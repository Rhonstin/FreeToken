#!/usr/bin/env bash
# 34 (17b): prompt lookup (draft-free speculation) A/B vs plain decode.
# Repetitive prompts should speed up; prose should be a no-op; greedy must be bit-identical.
set -uo pipefail
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"; export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
M=/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4
RAW=/opt/FreeToken/evidence/x299/34/raw
P="$RAW/prompts"
PY=/opt/freetoken-venv/bin/python
cd /opt/FreeToken; mkdir -p "$P"
restart(){ echo "[34] restart prod"; sudo -n systemctl start freetoken.service; }
trap restart EXIT
# fixtures
"$PY" - <<'PYEOF'
import json,pathlib
d=pathlib.Path("/opt/FreeToken/evidence/x299/34/raw/prompts"); d.mkdir(parents=True,exist_ok=True)
w=("The origin of the modern weather forecast lies in the nineteenth century when a severe storm struck "
   "the British Isles and observers began sending simultaneous reports by telegraph from many stations. ")
prose=json.loads(open("/opt/FreeToken/evidence/x299/03b/fixtures/prompt_16k.jsonl").readline())["prompt"]
for name,unit,target in (("rep1k",w,4000),("rep4k",w,15000)):
    d.joinpath(f"prompt_{name}.jsonl").write_text(json.dumps({"prompt":(unit*((target//len(unit))+1))[:target]})+"\n")
d.joinpath("prompt_prose16k.jsonl").write_text(json.dumps({"prompt":prose})+"\n")
print("fixtures written")
PYEOF
sudo -n systemctl stop freetoken.service
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 500 ] && break; sleep 2; done
rm -f "$RAW/probe.jsonl" "$RAW/miss.txt"
run(){ local id=$1 port=$2 extra=$3
  echo "[34] $id $(date -u +%H:%M:%S)"
  ( setsid "$PY" -m freetoken.cli serve --model "$M" --host 127.0.0.1 --port "$port" \
      --moe-strategy offload --moe-cache-auto --kv-cache-dtype nvfp4 \
      --num-tokens 220032 --kv-reserve-tokens 220032 --memory-ratio 0.90 \
      --moe-collect-stats --ple-backend disk --max-running-requests 1 --cuda-graph-max-bs 1 $extra \
      > "$RAW/serve-$id.log" 2>&1 < /dev/null & )
  ok=0; for i in $(seq 1 180); do curl -s -m5 "localhost:$port/health" 2>/dev/null | grep -q '"serving"' && { ok=1; break; }; sleep 5; done
  curl -s -m180 "localhost:$port/v1/models" > "$RAW/models-$id.json" 2>/dev/null
  if [ "$ok" = 1 ]; then
    "$PY" evidence/x299/05b/context_probe.py --origin "http://127.0.0.1:$port" --prompt-dir "$P" \
      --contexts rep1k,rep4k,prose16k --decode 128 --mode "$id" --jsonl "$RAW/probe.jsonl" > "$RAW/probe-$id.log" 2>&1
    echo "$id rc=$? vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$RAW/miss.txt"
  else echo "$id NOT READY" >> "$RAW/miss.txt"; fi
  pkill -f -- "--port $port" 2>/dev/null; sleep 4
}
run plain  2111 ""
run lookup 2112 "--lookup-draft 4 --lookup-ngram 6 --lookup-min-ngram 2"
echo "[34] done $(date -u +%H:%M:%S)"
