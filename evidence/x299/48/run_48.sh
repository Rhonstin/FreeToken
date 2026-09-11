#!/usr/bin/env bash
# 48: GSM8K greedy decode-quality A/B, offload vs hybrid fetch1, same auto@mr1 / 220k.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/48/raw
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[48] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/gsm8k-eval.jsonl"

run(){ local id=$1 port=$2 strat=$3 portflag=$4
  echo "[48] $id $strat $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy "$strat" $portflag --moe-cache-auto --kv-cache-dtype nvfp4 \
      --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
      --ple-backend disk --max-running-requests 1 \
      > "$RAW/serve-$id.log" 2>&1 < /dev/null & )
  ok=0; for i in $(seq 1 180); do curl -s -m5 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  curl -s -m180 "localhost:$port/v1/models" > "$RAW/models-$id.json" 2>/dev/null
  grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve-$id.log" | tail -1
  if [ "$ok" = 1 ]; then
    "$PY" evidence/x299/48/gsm8k_eval.py --origin "http://127.0.0.1:$port" --data "$RAW/gsm8k.jsonl" \
      --tag "gsm8k-$id" --out "$RAW/gsm8k-eval.jsonl" --max-tokens 1536 --effort low
  else echo "$id NOT READY"; fi
  pkill -f -- "--port $port" 2>/dev/null; sleep 4
}

run offload 2096 offload ""
run hybrid  2097 hybrid "--moe-hybrid-max-fetch 1"
echo "[48] done $(date -u +%H:%M:%S)"
