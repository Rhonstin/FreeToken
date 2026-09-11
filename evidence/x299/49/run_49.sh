#!/usr/bin/env bash
# 49: validate hybrid fetch1 at the adopted prod shape (auto@mr1, 220k) before switching.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/49/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[49] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/probe.jsonl" "$RAW/miss.txt"
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 2101 \
    --moe-strategy hybrid --moe-hybrid-max-fetch 1 --moe-cache-auto --kv-cache-dtype nvfp4 \
    --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
    --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
    > "$RAW/serve-hybrid.log" 2>&1 < /dev/null & )
ok=0; for i in $(seq 1 180); do curl -s -m5 localhost:2101/health 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
curl -s -m180 localhost:2101/v1/models > "$RAW/models.json" 2>/dev/null
grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve-hybrid.log" | tail -1 > "$RAW/geom.txt"; cat "$RAW/geom.txt"
if [ "$ok" = 1 ]; then
  for ctx in 16k 64k; do
    "$PY" evidence/x299/05b/context_probe.py --origin http://127.0.0.1:2101 --prompt-dir "$FIX" \
      --contexts "$ctx" --decode 128 --mode "hybrid-f1-$ctx" --jsonl "$RAW/probe.jsonl" > "$RAW/probe-$ctx.log" 2>&1
    echo "hybrid-f1-$ctx rc=$? vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) $(grep -aoE 'moe miss: [0-9.]+' "$RAW/serve-hybrid.log" | tail -1)" >> "$RAW/miss.txt"
  done
  echo "[49] max prefill $(date -u +%H:%M:%S)"
  "$PY" evidence/x299/45/maxprefill_probe.py --origin http://127.0.0.1:2101 --out "$RAW/maxprefill.json" > "$RAW/maxprefill.log" 2>&1
  echo "maxprefill rc=$? vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$RAW/miss.txt"
  echo "[49] gsm8k sanity (10) $(date -u +%H:%M:%S)"
  head -10 evidence/x299/48/raw/gsm8k.jsonl > "$RAW/gsm8k10.jsonl"
  "$PY" evidence/x299/48/gsm8k_eval.py --origin http://127.0.0.1:2101 --data "$RAW/gsm8k10.jsonl" \
    --tag gsm8k-hybrid-prodshape --out "$RAW/gsm8k.jsonl" --max-tokens 1536 --effort low 2>&1 | tail -1
else echo "hybrid NOT READY"; tail -5 "$RAW/serve-hybrid.log"; fi
grep -aiE "out of memory|CUDA out of memory" "$RAW/serve-hybrid.log" | head -2 >> "$RAW/miss.txt"
pkill -f -- "--port 2101" 2>/dev/null
echo "[49] done $(date -u +%H:%M:%S)"
