#!/usr/bin/env bash
# 07b: expert-cache/KV split frontier. Stops prod, measures, ALWAYS restarts.
# Core test: with KV sized for 16k (num-tokens 32768), does more expert cache
# reduce misses and raise TPOT? Plus auto vs manual and a prefill-peak check.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/07b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[07b] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
echo "[07b] stop prod"; sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/split.jsonl" "$RAW/miss.txt"

run() {  # id port cache_arg numtokens ctx
  local id=$1 port=$2 cache=$3 nt=$4 ctx=$5
  echo "[07b] $id cache=$cache nt=$nt ctx=$ctx $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy offload $cache --kv-cache-dtype nvfp4 \
      --num-tokens "$nt" --kv-reserve-tokens "$nt" --memory-ratio 0.90 \
      --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      > "$RAW/serve-$id.log" 2>&1 < /dev/null & )
  local ok=0; for i in $(seq 1 120); do curl -s -m3 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  if [ "$ok" = 1 ]; then
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" \
      --prompt-dir "$FIX" --contexts "$ctx" --decode 128 --mode "$id" \
      --jsonl "$RAW/split.jsonl" > "$RAW/probe-$id.log" 2>&1
    local rc=$?
    local miss; miss=$(grep -aoE "moe miss: [0-9.]+" "$RAW/serve-$id.log" | tail -1)
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "$id $ctx rc=$rc vram_used_mib=$used $miss" >> "$RAW/miss.txt"
    echo "[07b] $id rc=$rc $miss vram=$used"
  else
    echo "$id NOT READY" >> "$RAW/miss.txt"; echo "[07b] $id NOT READY (OOM?)"
  fi
  grep -aiE "out of memory|OutOfMemory|CUDA out of memory|ValueError: cache budget" "$RAW/serve-$id.log" | head -2 >> "$RAW/miss.txt"
  pkill -f "freetoken.cli serve --model $M --host 127.0.0.1 --port $port" 2>/dev/null
  pkill -f "ft serve --model-path $M.*--port $port" 2>/dev/null
  sleep 4
}

# Context-appropriate KV (32768), expert cache curve.
run s2600 1961 "--moe-cache-size 2600" 32768 16k
run s3200 1962 "--moe-cache-size 3200" 32768 16k
run s4000 1963 "--moe-cache-size 4000" 32768 16k
# auto vs manual
run auto  1964 "--moe-cache-auto" 32768 16k
# prod reference (KV 220k)
run ref220k 1965 "--moe-cache-size 2600" 220032 16k
# prefill-peak at 64k with the largest cache that fit (change to s2600 if 4000 OOMed)
run prefill64k 1966 "--moe-cache-size 3200" 32768 64k

echo "[07b] done $(date -u +%H:%M:%S)"
