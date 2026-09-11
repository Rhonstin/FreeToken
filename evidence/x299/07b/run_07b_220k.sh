#!/usr/bin/env bash
# 07b follow-up: does --moe-cache-auto work with the 220k context (does it OOM)?
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/07b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[220k] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/auto220k.jsonl"
run(){ local id=$1 port=$2 ctx=$3
  echo "[220k] $id ctx=$ctx $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy offload --moe-cache-auto --kv-cache-dtype nvfp4 \
      --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
      --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      > "$RAW/serve-auto220k-$id.log" 2>&1 < /dev/null & )
  ok=0; for i in $(seq 1 150); do curl -s -m3 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  if [ "$ok" = 1 ]; then
    local geom; geom=$(grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve-auto220k-$id.log" | tail -1)
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" --prompt-dir "$FIX" \
      --contexts "$ctx" --decode 128 --mode "$id" --jsonl "$RAW/auto220k.jsonl" > "$RAW/probe-auto220k-$id.log" 2>&1
    echo "$id ctx=$ctx rc=$? vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) [$geom] $(grep -aoE 'moe miss: [0-9.]+' "$RAW/serve-auto220k-$id.log" | tail -1)" >> "$RAW/auto220k-fit.txt"
  else echo "$id NOT READY (OOM?)" >> "$RAW/auto220k-fit.txt"; fi
  pkill -f "freetoken.cli serve --model $M --host 127.0.0.1 --port $port" 2>/dev/null
  pkill -f "ft serve --model-path $M.*--port $port" 2>/dev/null; sleep 4
}
run auto220k-16k 1981 16k
run auto220k-64k 1982 64k
echo "[220k] done $(date -u +%H:%M:%S)"
