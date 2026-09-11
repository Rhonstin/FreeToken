#!/usr/bin/env bash
# 06b confirmation: ABBA t6/t8 at 4k for a stable thread recommendation.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/06b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/threads-confirm.jsonl"
i=0
for spec in 6 8 8 6; do
  i=$((i+1)); port=$((1950+i))
  echo "[conf] run$i threads=$spec $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy hybrid --moe-hybrid-max-fetch 1 --moe-cache-size 2600 \
      --moe-cpu-threads "$spec" --kv-cache-dtype nvfp4 --num-tokens 220032 --kv-reserve-tokens 220032 \
      --memory-ratio 0.90 --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      > "$RAW/serve-conf-$i.log" 2>&1 < /dev/null & )
  ok=0; for j in $(seq 1 120); do curl -s -m3 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  if [ "$ok" = 1 ]; then
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" --prompt-dir "$FIX" \
      --contexts 4k --decode 128 --mode "t$spec-run$i" --jsonl "$RAW/threads-confirm.jsonl" > "$RAW/probe-conf-$i.log" 2>&1
    echo "[conf] run$i rc=$?"
  else echo "[conf] run$i NOT READY"; fi
  pkill -f "freetoken.cli serve --model $M --host 127.0.0.1 --port $port" 2>/dev/null
  pkill -f "ft serve --model-path $M.*--port $port" 2>/dev/null
  sleep 4
done
echo "[conf] done $(date -u +%H:%M:%S)"
