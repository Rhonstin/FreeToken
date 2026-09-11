#!/usr/bin/env bash
# 03b: run the extended decode-latency benchmarks on llmserver. Stops prod, runs,
# ALWAYS restarts prod. Decode bench bs=1 + paired KV context sweep 4k/16k/32k/64k.
set -uo pipefail

export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/03b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
mkdir -p "$RAW"

restart() {
  echo "[03b] restarting freetoken.service"
  sudo -n systemctl start freetoken.service || echo "[03b] RESTART FAILED" >&2
}
trap restart EXIT

echo "[03b] stopping freetoken.service"
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  [ "$u" -lt 500 ] && { echo "[03b] vram free ($u MiB)"; break; }
  sleep 2
done

rm -f "$RAW/decode-rows.jsonl"
echo "[03b] bench_decode_moe start $(date -u +%H:%M:%S)"
$PY benchmarks/bench_decode_moe.py --model "$M" --backend offload --decode 256 --cache 2600 \
    --greedy --json "$RAW/decode-rows.jsonl" > "$RAW/decode.log" 2>&1
echo "[03b] decode rc=$? $(date -u +%H:%M:%S)"

rm -f "$RAW/kvq-rows.json" "$RAW/kvq-raw.jsonl"
echo "[03b] bench_kv_quant start $(date -u +%H:%M:%S)"
$PY benchmarks/bench_kv_quant.py --model "$M" --prompt-dir "$FIX" \
    --contexts 4k,16k,32k,64k --decode 128 --cache 2600 \
    --repeats-short 3 --repeats-long 1 --skip-modes fp8 \
    --json "$RAW/kvq-rows.json" --raw-jsonl "$RAW/kvq-raw.jsonl" > "$RAW/kvq.log" 2>&1
echo "[03b] kvq rc=$? $(date -u +%H:%M:%S)"

echo "[03b] done"
