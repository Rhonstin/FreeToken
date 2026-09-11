#!/usr/bin/env bash
# 15b: MTP cost under --moe-cache-auto (the only fitting mode) — depth0 vs depth1.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/15b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[15b2] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
probe(){ local d=$1 port=$2
  echo "[15b2] depth=$d auto $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy offload --moe-cache-auto --kv-cache-dtype nvfp4 \
      --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
      --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      --mtp-depth "$d" > "$RAW/serve-auto-d$d.log" 2>&1 < /dev/null & )
  local ok=0; for i in $(seq 1 100); do curl -s -m3 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  if [ "$ok" = 1 ]; then
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" --prompt-dir "$FIX" \
      --contexts 4k --decode 128 --mode "auto-depth$d" --jsonl "$RAW/mtp-auto.jsonl" > "$RAW/probe-auto-d$d.log" 2>&1
    local geom miss acc vram
    geom=$(grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \([^)]*\)" "$RAW/serve-auto-d$d.log" | tail -1)
    miss=$(grep -aoE "moe miss: [0-9.]+" "$RAW/serve-auto-d$d.log" | tail -1)
    acc=$(grep -aoE "spec accept: [0-9]+/[0-9]+ \([0-9.]+\)" "$RAW/serve-auto-d$d.log" | tail -1)
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "auto depth=$d rc=$? vram_mib=$vram $geom $miss $acc" >> "$RAW/mtp-auto-summary.txt"
    echo "[15b2] depth=$d $geom $miss $acc vram=$vram"
  else echo "auto depth=$d NOT READY" >> "$RAW/mtp-auto-summary.txt"; tail -3 "$RAW/serve-auto-d$d.log"; fi
  pkill -f -- "--port $port" 2>/dev/null; sleep 4
}
rm -f "$RAW/mtp-auto.jsonl" "$RAW/mtp-auto-summary.txt"
probe 0 2031
probe 1 2032
echo "[15b2] done $(date -u +%H:%M:%S)"
