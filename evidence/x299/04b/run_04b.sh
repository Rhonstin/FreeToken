#!/usr/bin/env bash
# 04b: nsys-trace one steady decode step (eager, kernels visible) + report.
# Stops prod, profiles bench_decode_moe (which manages its own server), restarts prod.
set -uo pipefail

export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/04b/raw
PY=/opt/freetoken-venv/bin/python
NSYS=/usr/local/bin/nsys
cd /opt/FreeToken
mkdir -p "$RAW"

restart() {
  echo "[04b] restarting freetoken.service"
  sudo -n systemctl start freetoken.service || echo "[04b] RESTART FAILED" >&2
}
trap restart EXIT

echo "[04b] stopping freetoken.service"
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 500 ] && { echo "[04b] vram free"; break; }
  sleep 2
done

rm -f "$RAW/decode-eager-rows.jsonl" "$RAW/decode_eager.nsys-rep"
echo "[04b] nsys profile start $(date -u +%H:%M:%S)"
"$NSYS" profile -o "$RAW/decode_eager" --force-overwrite=true \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  -- "$PY" benchmarks/bench_decode_moe.py --model "$M" --backend offload --decode 256 \
  --cache 2600 --greedy --no-graph --json "$RAW/decode-eager-rows.jsonl" > "$RAW/nsys-decode.log" 2>&1
echo "[04b] nsys profile rc=$? $(date -u +%H:%M:%S)"

echo "[04b] nsys stats"
"$NSYS" stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum \
  --format table "$RAW/decode_eager.nsys-rep" > "$RAW/nsys-stats.txt" 2>&1
echo "[04b] stats rc=$?"

echo "[04b] done $(date -u +%H:%M:%S)"
