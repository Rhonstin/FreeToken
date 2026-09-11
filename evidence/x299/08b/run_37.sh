#!/usr/bin/env bash
# #37: re-bench with reps and show the confidence fields.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/08b/raw
PY=/opt/freetoken-venv/bin/python
PROF="$HOME/.cache/freetoken/benchbw/GPU-fd98efd2-8559-aa03-d51e-99d494ae8061.json"
cd /opt/FreeToken; mkdir -p "$RAW"
restart(){ echo "[37] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
echo "[37] bench --reps 3 $(date -u +%H:%M:%S)"
"$PY" -m freetoken.cli bench bw --dtype nvfp4 --model qwen3.8-next --reps 3 --cpu-threads 6 > "$RAW/bench-reps.log" 2>&1
echo "[37] rc=$?"
grep -aE "nvfp4|runs:|overlapped|backend" "$RAW/bench-reps.log" | tail -12
"$PY" - "$PROF" <<'PY' > "$RAW/profile-reps.txt" 2>&1
import json,sys
d=json.load(open(sys.argv[1]))
e=(d.get("dtype_kernels") or {}).get("nvfp4",{})
print("dtype nvfp4:", {k:e.get(k) for k in ("recommended","confident","reps","ratio","ratio_range","cpu_moe_gbs","cpu_moe_gbs_runs","pcie_gather_gbs","pcie_gather_gbs_runs")})
for n,wl in (d.get("workloads") or {}).items():
    k=(wl.get("kernels") or {}).get("nvfp4",{})
    print(n, {kk:k.get(kk) for kk in ("recommended","confident","reps","ratio","ratio_range")})
PY
cat "$RAW/profile-reps.txt"
echo "[37] done $(date -u +%H:%M:%S)"
