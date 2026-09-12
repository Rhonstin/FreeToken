# FreeToken /metrics vs llama.cpp /metrics

`llama.cpp` (192.168.2.139:8081) exposes token counters, two throughput gauges,
queue depth and spec counters. `freetoken` exposes those plus every category the
dashboard needs, so the dashboard no longer needs ssh or engine-log parsing.

| category | llama.cpp | freetoken |
|---|---|---|
| prompt/generated token counters | prompt_tokens_total, prompt_tokens_cached_total, tokens_predicted_total | freetoken_prompt_tokens_total, freetoken_prompt_tokens_cached_total, freetoken_generated_tokens_total |
| throughput gauges | prompt_tokens_seconds, predicted_tokens_seconds | freetoken_tokens_per_second{phase=decode|prefill} |
| queue / in-flight | requests_processing, requests_deferred | freetoken_requests_active, freetoken_requests_queued |
| KV cache | (none) | freetoken_kv_pages{kind=used|total}, freetoken_kv_usage_ratio |
| GDN / mamba pool | (none) | freetoken_mamba_slots{kind=used|total} |
| latency percentiles | (none) | freetoken_ttft_milliseconds{mean,p50,p95}, freetoken_request_duration_milliseconds{p50,p95}, freetoken_decode_milliseconds_per_token{p50,p95} |
| GPU telemetry | (none) | freetoken_gpu_utilization_ratio, gpu_memory_used/total_bytes, gpu_temperature_celsius, gpu_power_watts (pynvml) |
| per-request prefill progress | (none) | freetoken_prompt_processed_tokens, freetoken_prompt_total_tokens, freetoken_prompt_usage_ratio, freetoken_prompt_eta_seconds |
| MoE / hybrid | (none) | freetoken_moe_cache_miss_ratio, moe_expert_cache_slots, moe_hybrid_fetch_fraction (--moe-collect-stats) |
| speculation | spec_decode_num_draft/accepted/drafts_total, per-position | freetoken_spec_draft_tokens_total, spec_accepted_tokens_total, spec_accept_ratio |
| misc | n_decode_total, n_tokens_max, n_busy_slots_per_decode | n_tokens_max, vram_bytes, uptime_seconds, freetoken_info{model,ctx,attn,moe} |

## Verdict

freetoken's set is a superset on the observability axes the dashboard needs (pools,
GPU, latency percentiles, queue, live prefill progress, MoE/hybrid). llama.cpp still
has a few counters freetoken does not yet mirror (wall-clock prompt/generation
`*_seconds_total`, `n_decode_total`, `n_busy_slots_per_decode`, per-position spec
accepts) -- optional follow-ups, not needed by the dashboard.
