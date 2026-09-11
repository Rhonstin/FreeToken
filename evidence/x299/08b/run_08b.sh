#!/usr/bin/env bash
# 08b end-to-end: benchbw writes a schema-5 profile; serve --moe-strategy auto picks hybrid.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/08b/raw
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[08b] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
echo "[08b] stop prod"; sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done

echo "[08b] benchbw writes v5 profile $(date -u +%H:%M:%S)"
"$PY" -m freetoken.cli bench bw --dtype nvfp4 --cpu-threads 6 > "$RAW/benchbw.log" 2>&1
echo "[08b] benchbw rc=$?"
PROF="$HOME/.cache/freetoken/benchbw/GPU-fd98efd2-8559-aa03-d51e-99d494ae8061.json"
"$PY" - "$PROF" <<'PY' > "$RAW/profile-v5.txt" 2>&1
import json,sys
d=json.load(open(sys.argv[1]))
print("schema_version", d.get("schema_version"))
print("has fingerprint", "fingerprint" in d)
fp=d.get("fingerprint",{})
print("cpu", {k:fp.get("cpu",{}).get(k) for k in ("family","model","stepping","isa","threads")})
print("ram", fp.get("ram"))
print("pcie", fp.get("pcie"))
print("gpu", fp.get("gpu"))
print("dtypes", d.get("dtypes"))
PY

echo "[08b] serve auto to check resolution $(date -u +%H:%M:%S)"
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 1999 \
    --moe-strategy auto --kv-cache-dtype nvfp4 --num-tokens 220032 --kv-reserve-tokens 220032 \
    --memory-ratio 0.90 --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
    > "$RAW/serve-auto.log" 2>&1 < /dev/null & )
res=""; for i in $(seq 1 60); do res=$(grep -aoE "Resolved config: moe_strategy='[a-z]+'" "$RAW/serve-auto.log" | tail -1); [ -n "$res" ] && break; sleep 3; done
echo "resolved: $res" > "$RAW/auto-resolution.txt"
grep -aoE "benchbw profile .* rejected: .*|moe_strategy='[a-z]+'" "$RAW/serve-auto.log" | tail -3 >> "$RAW/auto-resolution.txt"
cat "$RAW/auto-resolution.txt"
pkill -f "freetoken.cli serve --model $M --host 127.0.0.1 --port 1999" 2>/dev/null
pkill -f "ft serve --model-path $M.*--port 1999" 2>/dev/null
echo "[08b] done $(date -u +%H:%M:%S)"
