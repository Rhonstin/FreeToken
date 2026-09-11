#!/usr/bin/env bash
# 45 follow-up: 4 concurrent requests (mr=4) + shared-context probe.
# 16k x4 = 64k (fits); 64k x4 = 256k (exceeds the 220032-token shared KV pool).
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/45/raw
FIX=/opt/FreeToken/evidence/x299/03b/fixtures
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[mr4] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/parallel-n4.jsonl"
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 2091 \
    --moe-strategy offload --moe-cache-auto --kv-cache-dtype nvfp4 \
    --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
    --moe-collect-stats --ple-backend disk --max-running-requests 4 \
    > "$RAW/serve-mr4.log" 2>&1 < /dev/null & )
ok=0; for i in $(seq 1 180); do curl -s -m5 localhost:2091/health 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
curl -s -m180 localhost:2091/v1/models > "$RAW/models-mr4.json" 2>/dev/null
grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve-mr4.log" | tail -1 > "$RAW/geom-mr4.txt"
cat "$RAW/geom-mr4.txt"
probe(){ local n=$1 pf=$2 dec=$3 tag=$4
  "$PY" evidence/x299/45/parallel_probe.py --origin http://127.0.0.1:2091 --prompt-file "$pf" \
    --n "$n" --decode "$dec" --tag "$tag" --out "$RAW/parallel-n4.jsonl" 2>&1 | \
    python3 -c "import sys,json;d=json.loads(sys.stdin.read());print(d['tag'],'wall',d['wall_s'],'agg',d['aggregate_tok_s'],'ok',d['all_ok'],[(q.get('decode_tok_s'),q.get('output_sha1'),q.get('error')) for q in d['requests']])"
}
probe 1 "$FIX/prompt_16k.jsonl" 128 mr4-n1-16k
probe 4 "$FIX/prompt_16k.jsonl" 128 mr4-n4-16k
probe 2 "$FIX/prompt_64k.jsonl" 64 mr4-n2-64k
probe 4 "$FIX/prompt_64k.jsonl" 64 mr4-n4-64k
pkill -f -- "--port 2091" 2>/dev/null
echo "[mr4] done $(date -u +%H:%M:%S)"
