#!/usr/bin/env bash
# 45: adopt --moe-cache-auto at 220k. Measures prod(manual2600) 16k baseline,
# then auto@220k at 16k/64k and a full ~219k prefill + sustained decode.
# Always restarts prod (manual config) at exit; adoption is a separate explicit step.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/45/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[45] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
# --- 1. prod (manual 2600, running) 16k baseline ---
if curl -s -m5 localhost:1919/health 2>/dev/null | grep -q '"serving"'; then
  echo "[45] prod 16k baseline $(date -u +%H:%M:%S)"
  "$PY" evidence/x299/05b/context_probe.py --origin http://127.0.0.1:1919 --prompt-dir "$FIX" \
    --contexts 16k --decode 128 --mode prod2600-16k --jsonl "$RAW/auto220k.jsonl" \
    > "$RAW/probe-prod2600-16k.log" 2>&1
fi
# --- 2. auto @ 220k ---
echo "[45] stop prod"; sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 2062 \
    --moe-strategy offload --moe-cache-auto --kv-cache-dtype nvfp4 \
    --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
    --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
    > "$RAW/serve-auto220k.log" 2>&1 < /dev/null & )
# Readiness = /health "serving" (the frontend answers /v1/models before the backend
# is ready and returns 503 for completions). Then warm the lazy reasoning-effort
# profile so context_probe's short /v1/models timeout is not hit.
ok=0
for i in $(seq 1 180); do curl -s -m5 localhost:2062/health 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
if [ "$ok" != 1 ]; then echo "[45] auto NOT READY"; tail -5 "$RAW/serve-auto220k.log"; fi
curl -s -m180 localhost:2062/v1/models > "$RAW/models.json" 2>/dev/null
echo "[45] models warmed $(wc -c < "$RAW/models.json") bytes"
grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve-auto220k.log" | tail -1 > "$RAW/geom.txt"
cat "$RAW/geom.txt"
for ctx in 16k 64k; do
  "$PY" evidence/x299/05b/context_probe.py --origin http://127.0.0.1:2062 --prompt-dir "$FIX" \
    --contexts "$ctx" --decode 128 --mode "auto220k-$ctx" --jsonl "$RAW/auto220k.jsonl" \
    > "$RAW/probe-auto220k-$ctx.log" 2>&1
  echo "auto220k-$ctx rc=$? vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) $(grep -aoE 'moe miss: [0-9.]+' "$RAW/serve-auto220k.log" | tail -1)" >> "$RAW/miss.txt"
done
# --- 3. full declared-max prefill + sustained decode ---
echo "[45] max prefill $(date -u +%H:%M:%S)"
"$PY" evidence/x299/45/maxprefill_probe.py --origin http://127.0.0.1:2062 --out "$RAW/maxprefill.json" \
  > "$RAW/maxprefill.log" 2>&1
echo "maxprefill rc=$? peak_vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$RAW/miss.txt"
grep -aiE "out of memory|CUDA out of memory|too long|maximum context" "$RAW/serve-auto220k.log" | head -3 >> "$RAW/miss.txt"
pkill -f -- "--port 2062" 2>/dev/null
echo "[45] done $(date -u +%H:%M:%S)"
