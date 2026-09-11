#!/usr/bin/env bash
# 41: score the fixed corpus under hybrid MoE (fetch1), same flags as prod offload.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/41/raw
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[41] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
echo "[41] hybrid serve $(date -u +%H:%M:%S)"
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 2051 \
    --moe-strategy hybrid --moe-hybrid-max-fetch 1 --moe-cache-size 2600 --kv-cache-dtype nvfp4 \
    --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 --moe-collect-stats \
    --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
    > "$RAW/serve-hybrid.log" 2>&1 < /dev/null & )
ok=0; for i in $(seq 1 100); do curl -s -m3 localhost:2051/health 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
if [ "$ok" = 1 ]; then
  "$PY" evidence/x299/41/score_ppl.py --origin http://127.0.0.1:2051 --tag hybrid-f1 \
    --out evidence/x299/41/raw/ppl-hybrid.json 2>&1 | tail -10
else echo "[41] hybrid NOT READY"; tail -4 "$RAW/serve-hybrid.log"; fi
pkill -f -- "--port 2051" 2>/dev/null
echo "[41] done $(date -u +%H:%M:%S)"
