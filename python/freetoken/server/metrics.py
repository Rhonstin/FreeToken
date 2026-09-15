"""Prometheus /metrics for the served model.

``to_prometheus`` is a pure formatter over the ``/v1/stats`` document (plus optional
enrichment: live GPU telemetry, prefill progress, MoE/hybrid counters, spec accounting),
so it is unit-testable without a GPU or a scheduler. ``gpu_snapshot`` is the only
impure part: a short-TTL pynvml sampler, import-guarded so a host without pynvml (or
without a GPU) simply exports no ``freetoken_gpu_*`` samples.

The metric set is a strict superset of llama.cpp's ``/metrics``: token counters,
throughput gauges and queue depth are matched, and KV/mamba/SWA pools, latency
percentiles, GPU util/mem/temp/power, per-request prefill progress and speculation
accounting are added.
"""

from __future__ import annotations

import threading
import time
from typing import Any


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if v is None:
        return "NaN"
    return repr(float(v))


def _labels(d: dict[str, Any] | None) -> str:
    if not d:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in d.items() if v is not None)
    return "{" + inner + "}" if inner else ""


class _Out:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def metric(self, name: str, mtype: str, help_: str, samples: list[tuple]) -> None:
        """samples: list of (value, labels_dict|None)."""
        if not samples:
            return
        samples = [(v, l) for v, l in samples if v is not None]
        if not samples:
            return
        self.lines.append(f"# HELP {name} {help_}")
        self.lines.append(f"# TYPE {name} {mtype}")
        for value, labels in samples:
            self.lines.append(f"{name}{_labels(labels)} {_fmt(value)}")


def to_prometheus(doc: dict[str, Any]) -> str:
    """Render a /v1/stats-style doc (+ enrichment) as Prometheus text."""
    o = _Out()
    model = doc.get("model") or {}
    rq = doc.get("requests") or {}
    tp = doc.get("throughput") or {}
    kv = doc.get("kv") or {}
    mamba = doc.get("mamba") or {}
    swa = doc.get("swa") or {}
    prefill = doc.get("prefill") or {}
    moe = doc.get("moe") or {}
    spec = doc.get("spec") or {}

    o.metric("freetoken_info", "gauge", "Served model identity and config.",
             [(1, {"model": model.get("id"), "ctx": model.get("ctx"),
                   "attn": model.get("attn"), "moe": str(bool(model.get("moe"))).lower(),
                   "kv_quant": kv.get("quant") or (doc.get("kv_quant") or "")})])
    o.metric("freetoken_uptime_seconds", "gauge", "Seconds since the engine became ready.",
             [(doc.get("uptime_s"), None)])
    o.metric("freetoken_prompt_tokens_total", "counter",
             "Prompt tokens processed, excluding cache hits.", [(rq.get("prompt_tokens_total"), None)])
    o.metric("freetoken_prompt_tokens_cached_total", "counter",
             "Prompt tokens served from the prefix cache.", [(rq.get("cached_tokens_total"), None)])
    o.metric("freetoken_generated_tokens_total", "counter",
             "Generation tokens emitted.", [(rq.get("completion_tokens_total"), None)])
    o.metric("freetoken_requests_active", "gauge", "Requests currently in flight.",
             [(rq.get("active"), None)])
    o.metric("freetoken_requests_queued", "gauge", "Requests deferred in the queue.",
             [(rq.get("queued"), None)])
    o.metric("freetoken_requests_completed_total", "counter", "Requests finished.",
             [(rq.get("completed"), None)])
    o.metric("freetoken_tokens_per_second", "gauge", "Sliding-window throughput (tokens/s).",
             [(tp.get("decode_tps"), {"phase": "decode"}), (tp.get("prefill_tps"), {"phase": "prefill"})])
    o.metric("freetoken_ttft_milliseconds", "gauge", "Time to first token (ms).",
             [(rq.get("ttft_mean_ms"), {"kind": "mean"}),
              (rq.get("ttft_p50_ms"), {"kind": "p50"}),
              (rq.get("ttft_p95_ms"), {"kind": "p95"})])
    o.metric("freetoken_request_duration_milliseconds", "gauge", "Full request duration (ms).",
             [(rq.get("duration_p50_ms"), {"kind": "p50"}),
              (rq.get("duration_p95_ms"), {"kind": "p95"})])
    o.metric("freetoken_decode_milliseconds_per_token", "gauge", "Steady-state decode latency (ms/token).",
             [(rq.get("decode_ms_p50"), {"kind": "p50"}),
              (rq.get("decode_ms_p95"), {"kind": "p95"})])
    o.metric("freetoken_n_tokens_max", "gauge", "Largest observed sequence length.",
             [(rq.get("n_tokens_max"), None)])
    o.metric("freetoken_kv_pages", "gauge", "KV page-pool usage (pages).",
             [(kv.get("used_pages"), {"kind": "used"}), (kv.get("total_pages"), {"kind": "total"})])
    o.metric("freetoken_kv_usage_ratio", "gauge", "KV pool used/total.",
             [(kv.get("used_pages") / kv.get("total_pages") if kv.get("total_pages") else None, None)])
    o.metric("freetoken_mamba_slots", "gauge", "GDN state-slot usage.",
             [(mamba.get("used_slots"), {"kind": "used"}), (mamba.get("total_slots"), {"kind": "total"})])
    o.metric("freetoken_swa_tokens", "gauge", "Sliding-window pool usage in tokens.",
             [(swa.get("used_pages"), {"kind": "used"}), (swa.get("total_pages"), {"kind": "total"})])
    o.metric("freetoken_vram_bytes", "gauge", "GPU bytes reserved by the engine process.",
             [(doc.get("vram_bytes"), None)])
    o.metric("freetoken_prompt_processed_tokens", "gauge",
             "Prompt tokens processed for the active prefill.", [(prefill.get("processed_tokens"), None)])
    o.metric("freetoken_prompt_total_tokens", "gauge",
             "Total prompt tokens of the active prefill.", [(prefill.get("total_tokens"), None)])
    o.metric("freetoken_prompt_usage_ratio", "gauge", "Prefill progress in [0,1].",
             [(prefill.get("usage_ratio"), None)])
    o.metric("freetoken_prompt_eta_seconds", "gauge", "Estimated seconds to finish the prefill.",
             [(prefill.get("eta_seconds"), None)])
    o.metric("freetoken_moe_cache_miss_ratio", "gauge", "Expert-cache miss ratio (--moe-collect-stats).",
             [(moe.get("miss_ratio"), None)])
    o.metric("freetoken_moe_expert_cache_slots", "gauge", "Expert-cache slots (--moe-collect-stats).",
             [(moe.get("cache_size"), None)])
    o.metric("freetoken_moe_hybrid_fetch_fraction", "gauge", "Hybrid PCIe fetch fraction.",
             [(moe.get("hybrid_fetch_fraction"), None)])
    o.metric("freetoken_spec_draft_tokens_total", "counter", "Speculative draft tokens proposed.",
             [(spec.get("proposed"), None)])
    o.metric("freetoken_spec_accepted_tokens_total", "counter", "Speculative draft tokens accepted.",
             [(spec.get("accepted"), None)])
    o.metric("freetoken_spec_accept_ratio", "gauge", "Speculative acceptance rate.",
             [(spec.get("rate"), None)])
    dp = doc.get("data_parallel") or {}
    if dp:
        o.metric("freetoken_dp_engines", "gauge", "Configured DP engine count.", [(dp.get("size"), None)])
        o.metric("freetoken_dp_engines_serving", "gauge", "DP engines currently serving.",
                 [(dp.get("serving"), None)])
    engines = doc.get("engines") or []
    if engines:
        o.metric("freetoken_engine_up", "gauge", "1 when the DP engine is serving.",
                 [(1 if e.get("state") == "serving" else 0, {"engine": e.get("index")}) for e in engines])
        o.metric("freetoken_engine_active_requests", "gauge", "In-flight requests per engine.",
                 [((e.get("in_flight") if e.get("in_flight") is not None
                    else (e.get("requests") or {}).get("active")), {"engine": e.get("index")}) for e in engines])
        o.metric("freetoken_engine_queued_requests", "gauge", "Queued requests per engine.",
                 [((e.get("requests") or {}).get("queued"), {"engine": e.get("index")}) for e in engines])
        o.metric("freetoken_engine_completed_total", "counter", "Completed requests per engine.",
                 [((e.get("requests") or {}).get("completed"), {"engine": e.get("index")}) for e in engines])
        o.metric("freetoken_engine_decode_tps", "gauge", "Sliding-window decode tok/s per engine.",
                 [((e.get("throughput") or {}).get("decode_tps"), {"engine": e.get("index")}) for e in engines])
        o.metric("freetoken_engine_prefill_tps", "gauge", "Sliding-window prefill tok/s per engine.",
                 [((e.get("throughput") or {}).get("prefill_tps"), {"engine": e.get("index")}) for e in engines])
        o.metric("freetoken_engine_vram_bytes", "gauge", "VRAM reserved per engine.",
                 [(e.get("vram_bytes"), {"engine": e.get("index")}) for e in engines])
        o.metric("freetoken_engine_restarts_total", "counter", "DP engine group restarts.",
                 [(e.get("restarts"), {"engine": e.get("index")}) for e in engines])
        o.metric("freetoken_engine_gpu_memory_bytes", "gauge", "Per-engine GPU memory total (bytes).",
                 [(((e.get("gpu") or {}).get("total_bytes")), {"engine": e.get("index"),
                   "gpu": (e.get("gpu") or {}).get("uuid")}) for e in engines])
    for g in doc.get("gpu_live") or []:
        labels = {"index": g.get("index"), "uuid": g.get("uuid"), "name": g.get("name")}
        o.metric("freetoken_gpu_utilization_ratio", "gauge", "GPU SM utilization in [0,1].",
                 [(g.get("util_ratio"), labels)])
        o.metric("freetoken_gpu_memory_used_bytes", "gauge", "GPU memory used (bytes).",
                 [(g.get("mem_used_bytes"), labels)])
        o.metric("freetoken_gpu_memory_total_bytes", "gauge", "GPU memory total (bytes).",
                 [(g.get("mem_total_bytes"), labels)])
        o.metric("freetoken_gpu_temperature_celsius", "gauge", "GPU temperature (C).",
                 [(g.get("temperature_c"), labels)])
        o.metric("freetoken_gpu_power_watts", "gauge", "GPU power draw (W).",
                 [(g.get("power_w"), labels)])
    return "\n".join(o.lines) + "\n"


# --------------------------------------------------------------------- live GPU
_GPU_LOCK = threading.Lock()
_GPU_CACHE: dict[str, Any] = {"ts": 0.0, "rows": []}
_GPU_TTL = 1.0


def gpu_snapshot() -> list[dict[str, Any]]:
    """pynvml GPU telemetry, short-TTL cached; [] when pynvml/the GPU is unavailable."""
    now = time.monotonic()
    with _GPU_LOCK:
        if now - _GPU_CACHE["ts"] < _GPU_TTL:
            return _GPU_CACHE["rows"]
        rows: list[dict[str, Any]] = []
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            try:
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    name = pynvml.nvmlDeviceGetName(h)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", "replace")
                    try:
                        uuid = pynvml.nvmlDeviceGetUUID(h)
                        if isinstance(uuid, bytes):
                            uuid = uuid.decode("utf-8", "replace")
                    except Exception:
                        uuid = None
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    except Exception:
                        temp = None
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    except Exception:
                        power = None
                    rows.append({
                        "index": i, "name": name, "uuid": uuid,
                        "util_ratio": util.gpu / 100.0,
                        "mem_used_bytes": int(mem.used), "mem_total_bytes": int(mem.total),
                        "temperature_c": temp, "power_w": round(power, 1) if power is not None else None,
                    })
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception:
            rows = []
        _GPU_CACHE["ts"], _GPU_CACHE["rows"] = now, rows
        return rows
