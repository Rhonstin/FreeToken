# FP8 KV-cache baseline: RTX 3090, Qwen3.8-Flash-Next (bf16 KV)

Baseline for the KV-cache quantization plan (beads epic `FreeToken-mtp-5y2`, task 5y2.2).
Every number is a real measurement from the offload box; commands and prompts are below so
task 5y2.12 can reproduce them.

## Platform

| item | value |
|---|---|
| GPU | NVIDIA RTX 3090, sm_86, 24576 MiB, driver 610.43.02 (UUID GPU-fd98efd2-...) |
| CPU / RAM | Intel i7-7800X, 6c/12t / 99 GiB |
| Torch / CUDA / Triton | 2.11.0+cu130 / 13.0 / 3.6.0 (venv `/opt/freetoken-venv`) |
| Code | `3d919e9` (deployed tree, dirty with the local MTP work), harness `0cccc09` |
| Backend / strategy | `--backend offload` (experts cached on GPU, missed rows over PCIe) |
| Sampling | `--greedy` (temperature 0, top_p 1, top_k -1) overrides the checkpoint's sampled defaults |

## Checkpoint

`RadixArk/Qwen3.8-Flash-Next-NVFP4`, `config.json` sha256
`e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624`.

- `Qwen4ExpForConditionalGeneration`: 48 layers = 36 linear-attention (GDN) + 12 full
  attention (`full_attention_interval 4`), 24 heads / 2 KV heads, head_dim 256, hidden
  2560, 512 experts top-10, vocab 248320, max_position 262144, MTP layers 1.
- Quantization: NVFP4 (modelopt 0.46.0), group 16; attention, MTP, GDN, gate/shared
  experts and PLE excluded from weight quantization.
- KV storage here is the compute dtype (bfloat16). FP8 projection at head_dim 256:
  2 x 256 code bytes + 2 fp32 row scales = 520 B/token vs 1024 B/token bf16.

## Prompts

One JSONL per context on the box (`/tmp/kvbase/prompt_{4k,16k,32k}.jsonl`):
instruction `"Summarize the log entries below in five bullet points.\n\n"` followed by a
repeated passage, tokenized with the checkpoint tokenizer + chat template. Realized prompt
tokens: 4109 / 16383 / 32780 (targets 4096 / 16384 / 32768, all within 0.2%).

## Command

```sh
cd /opt/FreeToken
export CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python:. /opt/freetoken-venv/bin/python3 \
  benchmarks/bench_decode_moe.py --model /home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --aime /tmp/kvbase/prompt_<ctx>.jsonl --backend offload --greedy --decode 200 \
  --max-seq-len <ctx+512> --cache <1900|1400> --json /tmp/kvbase/results.json
```

## Results (bs=1, decode 200, warm run; cold = first request after load)

| context | graph | cache | decode tok/s | ms/token | TTFT cold/warm (ms) | event p50/p99 (ms) | VRAM (GiB) | output sha1 |
|---|---|---|---|---|---|---|---|---|
| 4k  | on  | 1900 | 15.77 | 63.4 | 8290 / 6237 | 62.7 / 93.8 | 21.66 | `252494f74382` |
| 16k | on  | 1900 | 14.75 | 67.8 | 17415 / 6782 | 67.0 / 123.0 | 22.60 | `23ad1c220444` |
| 32k | on  | 1400 | 13.28 | 75.3 | 34122 / 32313 | 75.3 / 117.6 | 22.73 | `09bfc29cae4f` |
| 4k  | off | 1900 | 13.29 | 75.2 | 10079 / 5672 | 72.7 / 100.8 | 21.56 | `252494f74382` |
| 16k | off | 1900 | 11.80 | 84.7 | 17956 / 5689 | 75.5 / 256.6 | 22.62 | `23ad1c220444` |
| 32k | off | 1400 | 8.96 | 111.6 | 35818 / 32510 | 107.6 / 299.4 | 22.67 | `09bfc29cae4f` |

Observations to carry into task 5y2.12:

- CUDA graph is worth 18-48% decode (15.77 vs 13.29 at 4k; 13.28 vs 8.96 at 32k); it is the
  first knob for every FP8 comparison.
- Decode degrades ~16% from 4k to 32k with the graph on (63.4 -> 75.3 ms/token): KV reads grow.
- The 32k warm TTFT does not drop (32.3 s ~= cold 34.1 s) while 4k/16k warm TTFT do
  (6.2/6.8 s): prefix reuse is not effective at 32k with this sizing. Investigate before
  claiming any long-context cache gain.
- `completion_tokens=199` for `--decode 200` in every run (198 measured steps); the harness
  warns. Ratios stay comparable, but task 5y2.12 must fix or document the off-by-one.
- GPU tests on this box are NOT skipped: `tests/kernels/test_triton_attention.py` and
  `tests/models/qwen4_exp/test_qsa_backend.py` = 45 passed, 0 skipped (56s).

## Model matrix status

- Qwen3.8-Flash-Next QSA (above): done.
- Small FULL model and a hybrid Qwen3.5/3.6 checkpoint: BLOCKED, neither exists on the box
  (only `RadixArk/Qwen3.8-Flash-Next-NVFP4` is present). Re-run these rows when a checkpoint
  is available; nothing in the FP8 plan depends on them except extra coverage.

## Independent oracles and quality gates (definitions; implementations land with their tasks)

1. Codec oracle (task 5y2.5/5y2.6): a reference E4M3 encode/decode in plain PyTorch, tested
   against the kernel for all 256 codes, boundaries (max normal, min subnormal, zero),
   underflow, ties, saturation and NaN/Inf inputs. Pass = bit-exact round trip on the code
   buffer and max relative error <= 1e-3 on dequantized values for random rows.
2. Dequantized-attention oracle (task 5y2.6/5y2.9): compute attention over dequantized fp8
   K/V rows with a plain PyTorch reference, on the same prompt/weights as the kernel path.
   Pass = output logits within the bf16 attention envelope: max abs error vs the bf16
   attention reference <= 2x the bf16-vs-fp32 reference error, and greedy token ids equal
   over the first 32 decode steps on the baseline prompts.
3. Quality set (task 5y2.12): needle-in-haystack retrieval over the 4k/16k/32k prompts plus
   long-context summarization NLL. Proposed thresholds before optimization: retrieval
   accuracy drop <= 1 point vs the bf16 baseline, mean NLL delta <= 0.02 nats/token, and
   greedy common-prefix ratio >= 0.9 vs the bf16 output over the first 128 tokens.
4. Capacity gate (task 5y2.12): measured KV bytes/token must equal the planner's
   `spec_kv_bytes_per_token` (520 B at head_dim 256) within pooling overhead, and the
   context that fits at a fixed VRAM budget must grow by >= 1.8x at fp8.
