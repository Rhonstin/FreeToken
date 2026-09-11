#!/usr/bin/env bash
# 14b A/B: default (auto, Triton marlin-style NVFP4 decode) vs --quant-backend moe.nvfp4=triton.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/14b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[14b3] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done

probe() { local label=$1 port=$2 qb=$3
  local extra=""; [ -n "$qb" ] && extra="--quant-backend $qb"
  echo "[14b3] $label qb=${qb:-auto} $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy offload --moe-cache-size 2600 --kv-cache-dtype nvfp4 \
      --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
      --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 $extra \
      > "$RAW/serve3-$label.log" 2>&1 < /dev/null & )
  local ok=0; for i in $(seq 1 100); do curl -s -m3 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  if [ "$ok" = 1 ]; then
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" --prompt-dir "$FIX" \
      --contexts 4k --decode 128 --mode "$label" --jsonl "$RAW/backend-ab2.jsonl" > "$RAW/probe3-$label.log" 2>&1
    echo "[14b3] $label probe rc=$?"
  else echo "[14b3] $label NOT READY"; tail -4 "$RAW/serve3-$label.log"; fi
  pkill -f -- "--port $port" 2>/dev/null; sleep 4
}
rm -f "$RAW/backend-ab2.jsonl"
probe auto 2011 ""
probe triton 2012 "moe.nvfp4=triton"
echo "[14b3] done $(date -u +%H:%M:%S)"
