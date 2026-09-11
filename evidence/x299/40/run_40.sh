#!/usr/bin/env bash
# 40 (20b): final regression - target suite, throttle check, rollback verify, re-adopt.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/40/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
D=/etc/systemd/system/freetoken.service.d/mtp-test.conf
BK=/opt/FreeToken/.x299-backup
cd /opt/FreeToken; mkdir -p "$RAW"
restore_adopted(){ echo "[40] ensure adopted config"; sudo -n cp "$BK/mtp-test.conf.before-40" "$D"; sudo -n systemctl daemon-reload; sudo -n systemctl restart freetoken.service; }
trap restore_adopted EXIT
sudo -n cp "$D" "$BK/mtp-test.conf.before-40"
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
# 1. target suite
P=""
for f in tests/moe/test_hybrid_fetch.py tests/moe/test_benchbw_verdict.py tests/moe/test_offload.py tests/engine/test_lookup_draft.py tests/kvcache tests/scheduler/test_scheduler_chunked_prefill.py; do [ -e "$f" ] && P="$P $f"; done
echo "[40] target suite:$P $(date -u +%H:%M:%S)"
"$PY" -m pytest -q $P -m "not slow" > "$RAW/pytest-target.txt" 2>&1
echo "suite rc=$? $(tail -1 "$RAW/pytest-target.txt")"
# 2. throttle check on the adopted config
echo "[40] throttle/server $(date -u +%H:%M:%S)"
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 2161 \
    --moe-strategy hybrid --moe-hybrid-max-fetch 1 --moe-cache-auto --kv-cache-dtype nvfp4 \
    --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
    --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
    > "$RAW/serve-throttle.log" 2>&1 < /dev/null & )
ok=0; for i in $(seq 1 180); do curl -s -m5 localhost:2161/health 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
curl -s -m180 localhost:2161/v1/models > /dev/null 2>&1
( for i in $(seq 1 400); do nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader; sleep 2; done > "$RAW/clocks.txt" ) &
CPID=$!
if [ "$ok" = 1 ]; then "$PY" evidence/x299/45/maxprefill_probe.py --origin http://127.0.0.1:2161 --out "$RAW/throttle-maxprefill.json" > "$RAW/throttle.log" 2>&1; fi
kill $CPID 2>/dev/null
echo "throttle: $(wc -l < "$RAW/clocks.txt") samples; sm_clock min/max $(awk -F, '{gsub(/ /,"",$1); if($1+0>0){if(m==""||$1<m)m=$1; if($1>M)M=$1}}END{print m"/"M}' "$RAW/clocks.txt")"
pkill -f -- "--port 2161" 2>/dev/null; sleep 2
# 3. rollback verify: baseline drop-in -> measure -> re-adopt -> measure
echo "[40] rollback -> baseline $(date -u +%H:%M:%S)"
sudo -n cp "$BK/mtp-test.conf.before-45" "$D"; sudo -n systemctl daemon-reload; sudo -n systemctl restart freetoken.service
for i in $(seq 1 180); do curl -s -m5 localhost:1919/health 2>/dev/null | grep -q '"serving"' && break; sleep 5; done
"$PY" evidence/x299/05b/context_probe.py --origin http://127.0.0.1:1919 --prompt-dir "$FIX" --contexts 16k --decode 128 --mode rollback-baseline-16k --jsonl "$RAW/rollback.jsonl" > "$RAW/rollback-baseline.log" 2>&1
echo "[40] re-adopt $(date -u +%H:%M:%S)"
restore_adopted
for i in $(seq 1 180); do curl -s -m5 localhost:1919/health 2>/dev/null | grep -q '"serving"' && break; sleep 5; done
"$PY" evidence/x299/05b/context_probe.py --origin http://127.0.0.1:1919 --prompt-dir "$FIX" --contexts 16k --decode 128 --mode readopted-16k --jsonl "$RAW/rollback.jsonl" > "$RAW/readopted.log" 2>&1
echo "[40] done $(date -u +%H:%M:%S)"
