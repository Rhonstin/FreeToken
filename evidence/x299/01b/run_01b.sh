#!/usr/bin/env bash
# 01b bandwidth bench on llmserver. Stops prod, measures, ALWAYS restarts prod.
# Run as the rhonstin user; uses passwordless sudo only for systemctl.
set -uo pipefail

# Mirror /usr/local/bin/freetoken-serve-exp: use the CUDA 13 toolkit shipped in the venv.
# The system /usr/bin/nvcc is 12.4 and makes the PCIe-gather kernel build fail against torch cu130.
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

OUT="$1"                       # e.g. /opt/FreeToken/evidence/x299/01b/raw/benchbw.json
DTYPES="${2:-nvfp4,bf16}"
RAW="$(dirname "$OUT")"
BASE="${OUT%.json}"
mkdir -p "$RAW"

restart() {
  echo "[01b] restarting freetoken.service"
  sudo -n systemctl start freetoken.service || echo "[01b] RESTART FAILED" >&2
}
trap restart EXIT

echo "[01b] stopping freetoken.service"
sudo -n systemctl stop freetoken.service

for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "[01b] vram_used_mib=$used"
  if [ "$used" -lt 500 ]; then break; fi
  sleep 2
done

nvidia-smi --query-gpu=name,memory.total,memory.used,pcie.link.gen.current,pcie.link.width.current,clocks.sm,power.draw,temperature.gpu --format=csv > "$BASE.idle-link.txt"

# Under-load PCIe sampler (1 Hz) -> proves negotiated gen/width during the bench.
( for i in $(seq 1 1800); do
    nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,utilization.gpu,power.draw,clocks.sm --format=csv,noheader
    sleep 1
  done ) > "$BASE.under-load-link.txt" &
SAMP=$!

echo "[01b] running ft bench bw --dtype $DTYPES"
/opt/freetoken-venv/bin/ft bench bw --dtype "$DTYPES" -o "$OUT" > "$BASE.log" 2>&1
RC=$?
echo "[01b] bench exit=$RC"

kill "$SAMP" 2>/dev/null
wait "$SAMP" 2>/dev/null

nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current --format=csv > "$BASE.post-link.txt"
echo "$RC" > "$BASE.exit"
exit "$RC"
