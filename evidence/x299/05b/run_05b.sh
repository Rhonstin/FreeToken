#!/usr/bin/env bash
# 05b: config sweep without new runtime code. Stops prod, runs candidates, restarts.
# C0 (offload graph1) and C6 (offload graph0) already exist in 03b/04b; this runs
# hybrid fetch {auto,-1->auto,0,1,2} and cpu. KV is the bench default here; the winner
# is re-verified prod-like (nvfp4) afterwards.
set -uo pipefail

export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/05b/raw
PY=/opt/freetoken-venv/bin/python
UUID=GPU-fd98efd2-8559-aa03-d51e-99d494ae8061
cd /opt/FreeToken
mkdir -p "$RAW"

restart(){ echo "[05b] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT

# Install the measured benchbw profile so hybrid auto-split uses the real 0.3192 fraction.
mkdir -p "$HOME/.cache/freetoken/benchbw"
cp -f /opt/FreeToken/evidence/x299/01b/raw/benchbw.json "$HOME/.cache/freetoken/benchbw/$UUID.json"
echo "[05b] installed benchbw profile for $UUID"

echo "[05b] stopping freetoken.service"
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done

run() {  # id backend fetch
  local id=$1 b=$2 f=$3
  echo "[05b] $id backend=$b fetch=$f start $(date -u +%H:%M:%S)"
  rm -f "$RAW/cand-$id.jsonl"
  timeout 1500 "$PY" benchmarks/bench_decode_moe.py --model "$M" --backend "$b" \
    --hybrid-fetch "$f" --decode 256 --cache 2600 --greedy \
    --json "$RAW/cand-$id.jsonl" > "$RAW/cand-$id.log" 2>&1
  echo "[05b] $id rc=$? $(date -u +%H:%M:%S)"
  local newest; newest=$(ls -t /tmp/bench-serve-*.log 2>/dev/null | head -1)
  [ -n "$newest" ] && { echo "### $id server=$newest"; grep -aE "ServerArgs\(|Resolved config|expert banks:|moe miss|fallback|OOM|out of memory" "$newest" | head -6; } > "$RAW/cand-$id.effective.txt"
}

run hybrid_auto hybrid -1
run hybrid_f0   hybrid 0
run hybrid_f1   hybrid 1
run hybrid_f2   hybrid 2
run cpu         cpu -1

echo "[05b] done $(date -u +%H:%M:%S)"
