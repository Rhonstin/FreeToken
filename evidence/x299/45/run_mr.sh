#!/usr/bin/env bash
# 45 follow-up: how much does --moe-cache-auto resolve at --max-running-requests 2?
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/45/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[mr] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/mr-sweep.jsonl" "$RAW/mr-sweep.txt"
run(){ local id=$1 port=$2 mr=$3
  echo "[mr] $id mr=$mr $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy offload --moe-cache-auto --kv-cache-dtype nvfp4 \
      --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
      --moe-collect-stats --ple-backend disk --max-running-requests "$mr" \
      > "$RAW/serve-mr-$id.log" 2>&1 < /dev/null & )
  ok=0; for i in $(seq 1 180); do curl -s -m5 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  curl -s -m180 "localhost:$port/v1/models" > "$RAW/models-mr-$id.json" 2>/dev/null
  if [ "$ok" = 1 ]; then
    local geom; geom=$(grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve-mr-$id.log" | tail -1)
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" --prompt-dir "$FIX" \
      --contexts 16k --decode 128 --mode "auto-mr$mr-16k" --jsonl "$RAW/mr-sweep.jsonl" > "$RAW/probe-mr-$id.log" 2>&1
    echo "$id mr=$mr rc=$? vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) [$geom]" >> "$RAW/mr-sweep.txt"
  else echo "$id NOT READY" >> "$RAW/mr-sweep.txt"; fi
  pkill -f -- "--port $port" 2>/dev/null; sleep 4
}
run m2 2081 2
echo "[mr] done $(date -u +%H:%M:%S)"
