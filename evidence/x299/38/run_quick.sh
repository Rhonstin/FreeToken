#!/usr/bin/env bash
# quick: diagnose 18 collection + rerun copy bench (22) with small slots.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[quick] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
echo "[quick] ls tests"
ls -la tests/moe/test_hybrid_fetch.py tests/moe/test_bench_profile.py
echo "[quick] 18 collect"
"$PY" -m pytest --collect-only -q tests/moe/test_hybrid_fetch.py tests/moe/test_bench_profile.py > evidence/x299/18/raw/pytest-target.txt 2>&1
echo "collect rc=$?"; tail -12 evidence/x299/18/raw/pytest-target.txt
echo "[quick] 18 run"
"$PY" -m pytest -q tests/moe/test_hybrid_fetch.py tests/moe/test_bench_profile.py >> evidence/x299/18/raw/pytest-target.txt 2>&1
echo "run rc=$?"; tail -4 evidence/x299/18/raw/pytest-target.txt
echo "[quick] 22 copy small $(date -u +%H:%M:%S)"
"$PY" benchmarks/bench_offload_cache_copy.py --models glm4.7-nvfp4 --cache-slots 128 --batch-sizes 1 --miss-rates 0.25 --repeat 10 > evidence/x299/22/raw/bench-copy.txt 2>&1
echo "22 copy rc=$?"; tail -6 evidence/x299/22/raw/bench-copy.txt
echo "[quick] done $(date -u +%H:%M:%S)"
