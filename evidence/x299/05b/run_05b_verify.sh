#!/usr/bin/env bash
# 05b verify: (1) greedy text dumps offload vs hybrid-fetch1 for a quality/prefix check,
# (2) prod-like nvfp4 context sweep (4k/16k/32k) for offload and the hybrid winner.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/05b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[05b-verify] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT

echo "[05b-verify] stop prod"; sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done

# (1) greedy text dumps for prefix/answer comparison
for spec in "offload:-1" "hybrid:1"; do
  b=${spec%%:*}; f=${spec##*:}
  rm -f "$RAW/q-$b-$f.jsonl"
  echo "[05b-verify] dump $b fetch=$f $(date -u +%H:%M:%S)"
  timeout 1200 "$PY" benchmarks/bench_decode_moe.py --model "$M" --backend "$b" --hybrid-fetch "$f" \
    --decode 256 --cache 2600 --greedy --dump-text "$RAW/text-$b-$f.txt" \
    --json "$RAW/q-$b-$f.jsonl" > "$RAW/q-$b-$f.log" 2>&1
  echo "[05b-verify] dump $b rc=$?"
done

serve() {  # strategy fetch port
  sudo -n systemctl stop freetoken.service 2>/dev/null
  ( setsid /opt/freetoken-venv/bin/ft serve --model "$M" --host 127.0.0.1 --port "$3" \
      --moe-strategy "$1" --moe-hybrid-max-fetch "$2" --moe-cache-size 2600 \
      --kv-cache-dtype nvfp4 --num-tokens 220032 --kv-reserve-tokens 220032 \
      --memory-ratio 0.90 --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      > "$RAW/serve-$1-$2.log" 2>&1 < /dev/null & )
}
wait_ready() { for i in $(seq 1 120); do curl -s -m3 "localhost:$1/health" 2>/dev/null | grep -q '"serving"' && return 0; sleep 5; done; return 1; }
kill_serve() { pkill -f "ft serve --model $M --host 127.0.0.1 --port $1" 2>/dev/null; sleep 3; }

for spec in "offload:-1:1931" "hybrid:1:1932"; do
  b=${spec%%:*}; rest=${spec#*:}; f=${rest%%:*}; port=${rest##*:}
  rm -f "$RAW/ctx-$b-$f.jsonl"
  echo "[05b-verify] ctx serve $b fetch=$f port=$port $(date -u +%H:%M:%S)"
  serve "$b" "$f" "$port"
  if wait_ready "$port"; then
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" \
      --prompt-dir "$FIX" --contexts 4k,16k,32k --decode 128 --mode "$b-f$f" \
      --jsonl "$RAW/ctx-$b-$f.jsonl" > "$RAW/ctx-$b-$f.log" 2>&1
    echo "[05b-verify] ctx $b rc=$?"
  else
    echo "[05b-verify] ctx $b SERVER NOT READY"
  fi
  kill_serve "$port"
done

echo "[05b-verify] done $(date -u +%H:%M:%S)"
