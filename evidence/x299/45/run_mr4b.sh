#!/usr/bin/env bash
# 45 follow-up: 4 DISTINCT concurrent requests at 16k/48k/60k to see if the single
# 220032-token KV pool is shared (4x48k=192k fits; 4x60k=240k must not).
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/45/raw
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken
restart(){ echo "[mr4b] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/parallel-distinct.jsonl"
"$PY" evidence/x299/45/gen_distinct_prompts.py 16000 4 "$RAW/p4_16k.jsonl"
"$PY" evidence/x299/45/gen_distinct_prompts.py 48000 4 "$RAW/p4_48k.jsonl"
"$PY" evidence/x299/45/gen_distinct_prompts.py 60000 4 "$RAW/p4_60k.jsonl"
( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port 2092 \
    --moe-strategy offload --moe-cache-auto --kv-cache-dtype nvfp4 \
    --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
    --moe-collect-stats --ple-backend disk --max-running-requests 4 \
    > "$RAW/serve-mr4b.log" 2>&1 < /dev/null & )
ok=0; for i in $(seq 1 180); do curl -s -m5 localhost:2092/health 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
curl -s -m180 localhost:2092/v1/models > "$RAW/models-mr4b.json" 2>/dev/null
grep -aoE "resolved moe_cache_size=[0-9]+ num_pages=[0-9]+ \(prefill_overlap=[A-Za-z]+\)" "$RAW/serve-mr4b.log" | tail -1 > "$RAW/geom-mr4b.txt"
cat "$RAW/geom-mr4b.txt"
probe(){ local n=$1 pf=$2 dec=$3 tag=$4
  "$PY" evidence/x299/45/parallel_probe.py --origin http://127.0.0.1:2092 --prompt-file "$pf" \
    --n "$n" --decode "$dec" --tag "$tag" --out "$RAW/parallel-distinct.jsonl" 2>&1 | \
    python3 -c "import sys,json;d=json.loads(sys.stdin.read());print(d['tag'],'wall',d['wall_s'],'agg',d['aggregate_tok_s'],'ok',d['all_ok'],[(q.get('decode_tok_s'),q.get('error')) for q in d['requests']])"
}
# warm up so the first measured run is not dominated by compile/capture
"$PY" evidence/x299/45/parallel_probe.py --origin http://127.0.0.1:2092 --prompt-file "$RAW/p4_16k.jsonl" --n 1 --decode 16 --tag warm --out "$RAW/warm-distinct.jsonl" >/dev/null 2>&1
probe 1 "$RAW/p4_16k.jsonl" 64 mr4b-n1-16k
probe 4 "$RAW/p4_16k.jsonl" 64 mr4b-n4-16k
probe 4 "$RAW/p4_48k.jsonl" 64 mr4b-n4-48k
probe 4 "$RAW/p4_60k.jsonl" 64 mr4b-n4-60k
pkill -f -- "--port 2092" 2>/dev/null
echo "[mr4b] done $(date -u +%H:%M:%S)"
