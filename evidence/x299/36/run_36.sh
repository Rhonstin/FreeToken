#!/usr/bin/env bash
# 36 (18b): joint KV quantization x expert residency.
#   A same-slots (cache 2600) isolates the codec; B same-VRAM (auto) captures residency.
# Also runs the 22 (11b) copy microbench on the idle GPU first.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/36/raw
RAW22=/opt/FreeToken/evidence/x299/22/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW" "$RAW22"
restart(){ echo "[36] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/probe.jsonl" "$RAW/summary.txt"
echo "[36] copy microbench (22) $(date -u +%H:%M:%S)"
"$PY" benchmarks/bench_offload_cache_copy.py > "$RAW22/bench-copy.txt" 2>&1; echo "copy rc=$?" >> "$RAW/summary.txt"
run(){ local id=$1 port=$2 dtype=$3 cflag=$4
  echo "[36] $id dtype=$dtype $cflag $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy offload $cflag --kv-cache-dtype "$dtype" \
      --num-tokens 32768 --kv-reserve-tokens 32768 --memory-ratio 0.90 \
      --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
      > "$RAW/serve-$id.log" 2>&1 < /dev/null & )
  ok=0; for i in $(seq 1 180); do curl -s -m5 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  curl -s -m180 "localhost:$port/v1/models" > "$RAW/models-$id.json" 2>/dev/null
  if [ "$ok" = 1 ]; then
    local kv; kv=$(grep -aoE "K \+ V = [0-9.]+ GiB" "$RAW/serve-$id.log" | tail -1)
    local geom; geom=$(grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve-$id.log" | tail -1)
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" --prompt-dir "$FIX" \
      --contexts 16k --decode 128 --mode "$id" --jsonl "$RAW/probe.jsonl" > "$RAW/probe-$id.log" 2>&1
    "$PY" evidence/x299/41/score_ppl.py --origin "http://127.0.0.1:$port" --corpus evidence/x299/41/corpus.jsonl \
      --tag "$id" --out "$RAW/ppl-$id.json" > "$RAW/ppl-$id.log" 2>&1
    echo "$id dtype=$dtype $cflag rc=$? vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) [$kv] [$geom] $(grep -aoE 'moe miss: [0-9.]+' "$RAW/serve-$id.log" | tail -1)" >> "$RAW/summary.txt"
  else echo "$id NOT READY (OOM?)" >> "$RAW/summary.txt"; fi
  grep -aiE "out of memory|CUDA out of memory" "$RAW/serve-$id.log" | head -1 >> "$RAW/summary.txt"
  pkill -f -- "--port $port" 2>/dev/null; sleep 4
}
# A: same slots 2600
run A-nvfp4 2141 nvfp4 "--moe-cache-size 2600"
run A-fp8   2142 fp8   "--moe-cache-size 2600"
run A-bf16  2143 bf16  "--moe-cache-size 2600"
# B: same VRAM (auto)
run B-nvfp4 2144 nvfp4 "--moe-cache-auto"
run B-fp8   2145 fp8   "--moe-cache-auto"
run B-bf16  2146 bf16  "--moe-cache-auto"
echo "[36] done $(date -u +%H:%M:%S)"
