#!/usr/bin/env bash
# 07b confirmation: repeat auto and s3200 (noise), and a valid 64k prefill at prod geometry.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/07b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[07b2] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/split2.jsonl"
run(){ local id=$1 port=$2 cache=$3 nt=$4 ctx=$5
  echo "[07b2] $id cache=$cache nt=$nt ctx=$ctx $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy offload $cache --kv-cache-dtype nvfp4 --num-tokens "$nt" --kv-reserve-tokens "$nt" \
      --memory-ratio 0.90 --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      > "$RAW/serve2-$id.log" 2>&1 < /dev/null & )
  ok=0; for i in $(seq 1 120); do curl -s -m3 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  if [ "$ok" = 1 ]; then
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" --prompt-dir "$FIX" \
      --contexts "$ctx" --decode 128 --mode "$id" --jsonl "$RAW/split2.jsonl" > "$RAW/probe2-$id.log" 2>&1
    echo "$id rc=$? $(grep -aoE 'moe miss: [0-9.]+' "$RAW/serve2-$id.log" | tail -1) vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$RAW/miss2.txt"
  else echo "$id NOT READY" >> "$RAW/miss2.txt"; fi
  pkill -f "freetoken.cli serve --model $M --host 127.0.0.1 --port $port" 2>/dev/null
  pkill -f "ft serve --model-path $M.*--port $port" 2>/dev/null; sleep 4
}
run auto-b 1971 "--moe-cache-auto" 32768 16k
run s3200-b 1972 "--moe-cache-size 3200" 32768 16k
run prefill64k-prod 1973 "--moe-cache-size 2600" 220032 64k
echo "[07b2] done $(date -u +%H:%M:%S)"
