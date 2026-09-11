#!/usr/bin/env bash
# 08 upstream-align: bench the served geometry (qwen3.8-next) and verify auto picks hybrid
# via the exact-geometry entry (not the dtype entry, >2x away).
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/08b/raw
PY=/opt/freetoken-venv/bin/python
PROF="$HOME/.cache/freetoken/benchbw/GPU-fd98efd2-8559-aa03-d51e-99d494ae8061.json"
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[08u] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
echo "[08u] stop prod"; sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done

echo "[08u] bench --dtype nvfp4 --model qwen3.8-next $(date -u +%H:%M:%S)"
"$PY" -m freetoken.cli bench bw --dtype nvfp4 --model qwen3.8-next --cpu-threads 6 > "$RAW/bench-permodel.log" 2>&1
echo "[08u] bench rc=$?"
"$PY" - "$PROF" <<'PY' > "$RAW/profile-permodel.txt" 2>&1
import json,sys
d=json.load(open(sys.argv[1]))
print("dtypes", d.get("dtypes"))
print("dtype nvfp4 expert_bytes", (d.get("dtype_kernels") or {}).get("nvfp4",{}).get("expert_bytes"))
for name,wl in (d.get("workloads") or {}).items():
    print("workload", name, "model", wl.get("model"), "recommended", (wl.get("kernels") or {}).get("nvfp4",{}).get("recommended"),
          "expert_bytes", (wl.get("kernels") or {}).get("nvfp4",{}).get("expert_bytes"))
PY
cat "$RAW/profile-permodel.txt"

echo "[08u] serve auto $(date -u +%H:%M:%S)"
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 1999 \
    --moe-strategy auto --kv-cache-dtype nvfp4 --num-tokens 220032 --kv-reserve-tokens 220032 \
    --memory-ratio 0.90 --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
    > "$RAW/serve-auto-permodel.log" 2>&1 < /dev/null & )
for i in $(seq 1 60); do grep -aq "Resolved config" "$RAW/serve-auto-permodel.log" && break; sleep 3; done
grep -aoE "Resolved config: moe_strategy='[a-z]+'" "$RAW/serve-auto-permodel.log" | tail -1 > "$RAW/auto-permodel.txt"
grep -aoE "benchbw profile.*(rejected|not applying).*|moe_strategy='[a-z]+'" "$RAW/serve-auto-permodel.log" | tail -3 >> "$RAW/auto-permodel.txt"
cat "$RAW/auto-permodel.txt"
pkill -f "freetoken.cli serve --model $M --host 127.0.0.1 --port 1999" 2>/dev/null
pkill -f "ft serve --model-path $M.*--port 1999" 2>/dev/null
echo "[08u] done $(date -u +%H:%M:%S)"
