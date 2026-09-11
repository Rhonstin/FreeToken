#!/usr/bin/env bash
# Target-side regression tests for the b-tasks 18/20/26/38 + TTFT probe for 38.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[bmisc] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
mkdir -p evidence/x299/18/raw evidence/x299/20/raw evidence/x299/26/raw evidence/x299/38/raw
echo "[bmisc] copy microbench (22) $(date -u +%H:%M:%S)"
"$PY" benchmarks/bench_offload_cache_copy.py --models glm4.7-nvfp4 --cache-slots 2600 --batch-sizes 1 --miss-rates 0.25 > evidence/x299/22/raw/bench-copy.txt 2>&1
echo "22 copy rc=$? $(tail -1 evidence/x299/22/raw/bench-copy.txt)"
echo "[bmisc] target pytest $(date -u +%H:%M:%S)"
"$PY" -m pytest -q tests/moe/test_hybrid_fetch.py tests/moe/test_bench_profile.py > evidence/x299/18/raw/pytest-target.txt 2>&1
echo "18 pytest rc=$? $(tail -1 evidence/x299/18/raw/pytest-target.txt)"
"$PY" -m pytest -q tests/moe/test_offload.py > evidence/x299/20/raw/pytest-target.txt 2>&1
echo "20 pytest rc=$? $(tail -1 evidence/x299/20/raw/pytest-target.txt)"
"$PY" -m pytest -q tests/models/qwen4_exp/test_ple_disk.py > evidence/x299/26/raw/pytest-target.txt 2>&1
echo "26 pytest rc=$? $(tail -1 evidence/x299/26/raw/pytest-target.txt)"
# TTFT probe on a server
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 2151 \
    --moe-strategy offload --moe-cache-auto --kv-cache-dtype nvfp4 \
    --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
    --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 \
    > evidence/x299/38/raw/serve.log 2>&1 < /dev/null & )
ok=0; for i in $(seq 1 180); do curl -s -m5 localhost:2151/health 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
curl -s -m180 localhost:2151/v1/models > /dev/null 2>&1
if [ "$ok" = 1 ]; then
  "$PY" evidence/x299/38/ttft_probe.py --origin http://127.0.0.1:2151 --out evidence/x299/38/raw/ttft.json 2>&1 | tail -4
else echo "38 server NOT READY"; fi
pkill -f -- "--port 2151" 2>/dev/null
echo "[bmisc] done $(date -u +%H:%M:%S)"
