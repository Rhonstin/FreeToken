#!/usr/bin/env bash
# 05b context sweep only (nvfp4, prod-like), offload vs hybrid-fetch1, 4k/16k/32k.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/05b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[ctx] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
echo "[ctx] stop prod"; sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done

for spec in "offload:-1:1931" "hybrid:1:1932"; do
  b=${spec%%:*}; rest=${spec#*:}; f=${rest%%:*}; port=${rest##*:}
  rm -f "$RAW/ctx-$b-$f.jsonl"
  echo "[ctx] serve $b fetch=$f port=$port $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy "$b" --moe-hybrid-max-fetch "$f" --moe-cache-size 2600 \
      --kv-cache-dtype nvfp4 --num-tokens 220032 --kv-reserve-tokens 220032 \
      --memory-ratio 0.90 --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      > "$RAW/serve2-$b-$f.log" 2>&1 < /dev/null & )
  ok=0; for i in $(seq 1 120); do curl -s -m3 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  if [ "$ok" = 1 ]; then
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" \
      --prompt-dir "$FIX" --contexts 4k,16k,32k --decode 128 --mode "$b-f$f" \
      --jsonl "$RAW/ctx-$b-$f.jsonl" > "$RAW/ctx2-$b-$f.log" 2>&1
    echo "[ctx] probe $b rc=$?"
  else
    echo "[ctx] serve $b NOT READY"
  fi
  pkill -f "freetoken.cli serve --model $M --host 127.0.0.1 --port $port" 2>/dev/null
  pkill -f "ft serve --model-path $M.*--port $port" 2>/dev/null
  sleep 4
done
echo "[ctx] done $(date -u +%H:%M:%S)"
