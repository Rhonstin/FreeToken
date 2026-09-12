#!/usr/bin/env bash
# Final adopted-prod measurement: prefill (TTFT/throughput) and generation (decode tok/s, p50/p95).
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
RAW=/opt/FreeToken/evidence/x299/final/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW"
{
  echo "utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "drop_in: $(grep -a ExecStart /etc/systemd/system/freetoken.service.d/mtp-test.conf | tail -1)"
  echo "gpu:"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "model:"; curl -s -m5 localhost:1919/v1/models | /opt/freetoken-venv/bin/python -c "import sys,json;d=json.load(sys.stdin);print(d['data'][0]['id'])" 2>/dev/null
  echo "kv:"; curl -s -m5 localhost:1919/v1/stats 2>/dev/null | head -c 300; echo
} > "$RAW/snapshot.txt" 2>&1
echo "[final] prefill/decode sweep $(date -u +%H:%M:%S)"
"$PY" evidence/x299/05b/context_probe.py --origin http://127.0.0.1:1919 --prompt-dir "$FIX" \
  --contexts 4k,16k,64k,192k --decode 64 --mode final-adopted --jsonl "$RAW/final.jsonl" \
  > "$RAW/sweep.log" 2>&1
echo "sweep rc=$?"
echo "[final] short-prompt generation $(date -u +%H:%M:%S)"
"$PY" evidence/x299/38/ttft_probe.py --origin http://127.0.0.1:1919 --out "$RAW/short.json" \
  > "$RAW/short.log" 2>&1
echo "short rc=$?"
echo "[final] done $(date -u +%H:%M:%S)"
