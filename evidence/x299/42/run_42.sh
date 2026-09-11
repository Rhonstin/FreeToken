#!/usr/bin/env bash
# 42: decode concurrency p50/p95 at n=1/2/4 on the adopted hybrid config (mr=4 test server).
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/42/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[42] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/parallel.jsonl"
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 2171 \
    --moe-strategy hybrid --moe-hybrid-max-fetch 1 --moe-cache-auto --kv-cache-dtype nvfp4 \
    --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
    --ple-backend disk --max-running-requests 4 \
    > "$RAW/serve.log" 2>&1 < /dev/null & )
ok=0; for i in $(seq 1 180); do curl -s -m5 localhost:2171/health 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
curl -s -m180 localhost:2171/v1/models > /dev/null 2>&1
grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve.log" | tail -1 > "$RAW/geom.txt"; cat "$RAW/geom.txt"
if [ "$ok" = 1 ]; then
  "$PY" evidence/x299/45/parallel_probe.py --origin http://127.0.0.1:2171 --prompt-file "/opt/FreeToken/evidence/x299/45/raw/p4_16k.jsonl" --n 1 --decode 16 --tag warm --out "$RAW/parallel.jsonl" >/dev/null 2>&1
  for n in 1 2 4; do
    "$PY" evidence/x299/45/parallel_probe.py --origin http://127.0.0.1:2171 --prompt-file "/opt/FreeToken/evidence/x299/45/raw/p4_16k.jsonl" --n "$n" --decode 128 --tag "mr4-n$n" --out "$RAW/parallel.jsonl" 2>&1 | tail -1
  done
else echo "NOT READY"; fi
pkill -f -- "--port 2171" 2>/dev/null
echo "[42] done $(date -u +%H:%M:%S)"
