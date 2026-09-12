#!/usr/bin/env bash
# Final cold-prefill measurement: one DISTINCT prompt per size, no warmup -> true TTFT.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
RAW=/opt/FreeToken/evidence/x299/final/raw
PY=/opt/freetoken-venv/bin/python
GEN=/opt/FreeToken/evidence/x299/45/gen_distinct_prompts.py
PROBE=/opt/FreeToken/evidence/x299/45/parallel_probe.py
cd /opt/FreeToken
rm -f "$RAW/cold.jsonl"
for t in 4000 16000 64000 196000; do
  "$PY" "$GEN" "$t" 1 "$RAW/cold_$t.jsonl" >/dev/null 2>&1
  "$PY" "$PROBE" --origin http://127.0.0.1:1919 --prompt-file "$RAW/cold_$t.jsonl" \
    --n 1 --decode 64 --tag "cold-$t" --out "$RAW/cold.jsonl" 2>&1 | tail -1
done
echo "[cold] done $(date -u +%H:%M:%S)"
