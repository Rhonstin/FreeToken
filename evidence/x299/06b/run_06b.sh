#!/usr/bin/env bash
# 06b: CPU thread/affinity/ISA sweep. Stops prod, measures, ALWAYS restarts.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/06b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
mkdir -p "$RAW"
restart(){ echo "[06b] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT

echo "[06b] stop prod"; sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done

echo "[06b] benchbw ISA sweep $(date -u +%H:%M:%S)"
"$PY" -m freetoken.cli bench bw --dtype nvfp4 --isa all --cpu-threads 6 \
  -o "$RAW/benchbw-isa.json" > "$RAW/benchbw-isa.log" 2>&1
echo "[06b] benchbw rc=$?"

run_serve_probe() {  # label port threads isa(optional)
  local label=$1 port=$2 threads=$3 isa=${4:-}
  local env_isa=""
  [ -n "$isa" ] && env_isa="FREETOKEN_CPU_MOE_ISA=$isa"
  echo "[06b] $label threads=$threads isa=${isa:-auto} $(date -u +%H:%M:%S)"
  ( setsid env $env_isa "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy hybrid --moe-hybrid-max-fetch 1 --moe-cache-size 2600 \
      --moe-cpu-threads "$threads" --kv-cache-dtype nvfp4 \
      --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
      --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      > "$RAW/serve-$label.log" 2>&1 < /dev/null & )
  local ok=0; for i in $(seq 1 120); do curl -s -m3 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  if [ "$ok" = 1 ]; then
    ( for j in $(seq 1 240); do
        f=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | sort -n)
        echo "$(date +%s) min=$(echo "$f" | head -1) max=$(echo "$f" | tail -1)"
        sleep 1
      done ) > "$RAW/clocks-$label.txt" &
    local SAMP=$!
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" \
      --prompt-dir "$FIX" --contexts 4k --decode 128 --mode "$label" \
      --jsonl "$RAW/threads.jsonl" > "$RAW/probe-$label.log" 2>&1
    kill $SAMP 2>/dev/null
    echo "[06b] $label probe rc=$?"
  else
    echo "[06b] $label NOT READY"
  fi
  pkill -f "freetoken.cli serve --model $M --host 127.0.0.1 --port $port" 2>/dev/null
  pkill -f "ft serve --model-path $M.*--port $port" 2>/dev/null
  sleep 4
}

rm -f "$RAW/threads.jsonl" "$RAW"/clocks-*.txt
run_serve_probe t2 1941 2
run_serve_probe t4 1942 4
run_serve_probe t6 1943 6
run_serve_probe t8smt 1944 8
run_serve_probe t6-avx2 1945 6 avx2

echo "[06b] done $(date -u +%H:%M:%S)"
