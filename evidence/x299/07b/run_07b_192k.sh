#!/usr/bin/env bash
# Long-context check: generate a 192k-token prompt and probe auto@220k to see if
# the cap 3447 survives a near-220k prefill (the known OOM regime).
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/07b/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[192k] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT

"$PY" - <<PY
import json
from tokenizers import Tokenizer
M="$M"
tok=Tokenizer.from_file(f"{M}/tokenizer.json")
base=("A modern processor executes a program through a deep pipeline with caches, branch predictors and out-of-order execution. ")*20000
ids=tok.encode(base).ids
n=196608
rep=(ids*((n//len(ids))+1))[:n]
text=tok.decode(rep)+"\n\nQuestion: Summarize the passage above in several detailed paragraphs.\nAnswer:"
open("$FIX/prompt_192k.jsonl","w").write(json.dumps({"prompt":text})+"\n")
print("192k actual", len(tok.encode(text).ids))
PY

sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/auto220k-192k.jsonl"
echo "[192k] serve auto@220k $(date -u +%H:%M:%S)"
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 1991 \
    --moe-strategy offload --moe-cache-auto --kv-cache-dtype nvfp4 \
    --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
    --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
    > "$RAW/serve-auto220k-192k.log" 2>&1 < /dev/null & )
ok=0; for i in $(seq 1 150); do curl -s -m3 localhost:1991/health 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
if [ "$ok" = 1 ]; then
  geom=$(grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve-auto220k-192k.log" | tail -1)
  "$PY" evidence/x299/05b/context_probe.py --origin http://127.0.0.1:1991 --prompt-dir "$FIX" \
    --contexts 192k --decode 64 --mode auto220k-192k --jsonl "$RAW/auto220k-192k.jsonl" > "$RAW/probe-auto220k-192k.log" 2>&1
  echo "auto220k-192k rc=$? vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) [$geom] $(grep -aoE 'moe miss: [0-9.]+' "$RAW/serve-auto220k-192k.log" | tail -1)" >> "$RAW/auto220k-fit.txt"
else echo "auto220k-192k NOT READY (OOM?)" >> "$RAW/auto220k-fit.txt"; fi
grep -aiE "out of memory|OutOfMemory|dropped|exceeds" "$RAW/serve-auto220k-192k.log" | head -3 >> "$RAW/auto220k-fit.txt"
pkill -f "freetoken.cli serve --model $M --host 127.0.0.1 --port 1991" 2>/dev/null
pkill -f "ft serve --model-path $M.*--port 1991" 2>/dev/null
echo "[192k] done $(date -u +%H:%M:%S)"
