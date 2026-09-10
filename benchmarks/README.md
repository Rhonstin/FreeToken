# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

`--spec-depth 0,2` adds MTP speculative runs: one server per depth, acceptance parsed
from the spawned server's log, and a quality gate against the depth-0 greedy output
(``--min-prefix-ratio``). Example:

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload \
    --spec-depth 0,2 --greedy --decode 256 --cache 4000
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.

## KV-cache quantization campaign (fp8 vs bf16)

`bench_kv_quant.py` runs one server per (kv mode, graph, expert-cache mode) and reuses it
across the context matrix, so every bf16/fp8 pair shares weights, prompts, seeds and
sampling. Cold = first request after load; warm rows follow a warmup. Capacity is the
server's own `Allocating N tokens for KV cache` line; `--quality` adds needle retrieval at
three depths, a JSON/tool-call and a coding check (NLL/PPL is not available: the server
rejects `logprobs`).

Campaign of 2026-09-10, RTX 3090, Qwen3.8-Flash-Next-NVFP4, expert cache 2600, mem-ratio
0.90, max-seq-len 262144, greedy:

```sh
python benchmarks/bench_kv_quant.py --model <ckpt> --prompt-dir <prompts> \
  --contexts 4k,16k,32k,64k,128k --cache 2600 --mem-ratio 0.90 --quality \
  --json out.json                          # main matrix
python benchmarks/bench_kv_quant.py ... --parallel 4 --max-running 4 --contexts 4k,32k
python benchmarks/bench_kv_quant.py ... --no-graph --contexts 32k
python benchmarks/bench_kv_quant.py ... --cache 0 --num-tokens 36864 --contexts 32k
```

| context | decode bf16 (tok/s) | decode fp8 | delta | TTFT delta |
|---|---|---|---|---|
| 4k  | 16.80 | 17.02 | -1.3% | +0.0% |
| 16k | 17.01 | 16.88 | +0.8% | +0.2% |
| 32k | 16.94 | 17.01 | -0.5% | -0.2% |
| 64k | 16.44 | 16.12 | +1.9% | +0.8% |
| 128k | 15.85 | 15.88 | -0.2% | warm hit: 1.18 s -> 0.65 s |

- Capacity at a fixed ~6.76 GiB KV budget: **155,392 (bf16) vs 299,328 (fp8) tokens = 1.9x**.
- `--moe-cache-auto` with a pinned 36,864-token KV reserve: KV 0.87 -> 0.45 GiB, resolved
  expert cache **3772 -> 3895 slots (+3.3%)**, decode +1.5%, TTFT +6.6%.
- Graph off at 32k: fp8 decode +7.4% vs bf16 (both ~12-13 tok/s -- the graph itself is
  worth ~25%: 17 vs 13 tok/s).
- Quality: needle retrieval **9/9 in both modes** at 4k/16k/32k (drop 0 pp), coding check
  passes in both; the free-form JSON check fails in both (methodology -- use the tools API
  for a real tool-call check). NLL/PPL: BLOCKED, no logprob API.
- Thresholds (task 5y2.12): retrieval drop <= 2 pp PASS, decode slowdown <= 10% PASS at
  concurrency 1 (+7.4% worst graph-off), TTFT <= +15% PASS (<= +6.6%, auto arm).
  Concurrency 4 stays **experimental**: requests serialize behind long prefills and the
  per-stream rates are noisy, so the aggregate is not a threshold comparison.
- Finding: without `--num-tokens/--kv-reserve-tokens`, `--moe-cache-auto` budgets the
  expert cache from the whole pool budget and leaves only the default 8,192-token KV
  reserve, so long prompts are dropped ("Input sequence length ... exceeds"). Pin the KV
  reserve for long-context auto-cache runs.
