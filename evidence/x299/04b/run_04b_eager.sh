#!/usr/bin/env bash
# 04b overhead control: uninstrumented decode, graph OFF (vs 03b graph ON 19.63 tok/s).
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/04b/raw
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[04b] restart"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/decode-eager-plain-rows.jsonl"
"$PY" benchmarks/bench_decode_moe.py --model "$M" --backend offload --decode 256 --cache 2600 \
  --greedy --no-graph --json "$RAW/decode-eager-plain-rows.jsonl" > "$RAW/decode-eager-plain.log" 2>&1
echo "[04b] eager uninstrumented rc=$?"
