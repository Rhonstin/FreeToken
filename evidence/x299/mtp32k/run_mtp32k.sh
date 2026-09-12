#!/usr/bin/env bash
# MTP research Phase 0: does MTP depth1 pay at KV32768/cache2600 on the 3090?
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/mtp32k/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[mtp32k] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/probe.jsonl" "$RAW/summary.txt"
run(){ local id=$1 port=$2 depth=$3
  echo "[mtp32k] $id depth=$depth $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy offload --moe-cache-size 2600 --mtp-depth "$depth" --kv-cache-dtype nvfp4 \
      --num-tokens 32768 --kv-reserve-tokens 32768 --memory-ratio 0.90 \
      --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      > "$RAW/serve-$id.log" 2>&1 < /dev/null & )
  ok=0; for i in $(seq 1 180); do curl -s -m5 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  curl -s -m180 "localhost:$port/v1/models" > /dev/null 2>&1
  if [ "$ok" = 1 ]; then
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" --prompt-dir "$FIX" \
      --contexts 16k --decode 64 --mode "$id" --jsonl "$RAW/probe.jsonl" > "$RAW/probe-$id.log" 2>&1
    local acc; acc=$(grep -aoE "spec accept: [0-9]+/[0-9]+" "$RAW/serve-$id.log" | tail -1)
    echo "$id depth=$depth rc=$? vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) miss=$(grep -aoE 'moe miss: [0-9.]+' "$RAW/serve-$id.log" | tail -1) [$acc]" >> "$RAW/summary.txt"
  else echo "$id depth=$depth NOT READY (OOM?)" >> "$RAW/summary.txt"; fi
  grep -aiE "out of memory|CUDA out of memory" "$RAW/serve-$id.log" | head -1 >> "$RAW/summary.txt"
  pkill -f -- "--port $port" 2>/dev/null; sleep 4
}
run depth0 2191 0
run depth1 2192 1
echo "[mtp32k] done $(date -u +%H:%M:%S)"
