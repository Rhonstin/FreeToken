"""Runtime metrics for /v1/stats. The FrontendManager owns one StatsTracker and feeds it
every UserReply (the single chokepoint in listen()). kv/mamba/vram keep their last-known-value
semantics like ShellStats; throughput uses an independent sliding-window rate (NOT cumulative
average like tok_s), so idle polls decay to zero by wall clock."""

from __future__ import annotations

import time
from collections import deque
from typing import Any


class StatsTracker:
    def __init__(self, window_s: float = 5.0) -> None:
        self.window_s = window_s
        # maxlen bounds memory on the headless path: stale-sample eviction is poll-driven
        # (only _rate() trims to window_s), and clients that never hit /v1/stats (e.g.
        # codex/claude via /v1/chat/completions) would otherwise grow these unbounded.
        # 4096 is generous vs the sliding window's span at any realistic reply rate.
        self._decode: "deque[tuple[float, int]]" = deque(maxlen=4096)
        self._prefill: "deque[tuple[float, int]]" = deque(maxlen=4096)
        self._inflight: set[int] = set()
        # Requests for which an abort was dispatched but the scheduler's explicit terminal
        # acknowledgement has not arrived yet. They remain active until that barrier, while
        # any sampled tokens racing the abort continue to count toward lifetime totals.
        self._aborting: set[int] = set()
        self.completed = 0
        # Cumulative prompt/completion tokens since this process started (lifetime for THIS served
        # model). Exposed in /v1/stats so the desktop can diff consecutive polls into per-model
        # "cost saved by running locally" accounting. Monotonic; resets when the process restarts.
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self.cached_tokens_total = 0
        self.kv_used_pages = 0
        self.kv_total_pages = 0
        # Tokens per KV page as the ENGINE resolved it (a backend can override the CLI
        # --page-size, e.g. qsa_sparse forces 64); 0 until an engine reply reports it.
        self.kv_page_size = 0
        self.mamba_used_slots = 0
        self.mamba_total_slots = 0
        self.swa_used_tokens = 0
        self.swa_total_tokens = 0
        self.vram_bytes = 0
        self.queue_reqs = 0
        # Active-prefill progress + speculative/MoE readouts for /metrics.
        self.prefill_active = False
        self.prompt_processed = 0
        self.prompt_total = 0
        self.prefill_rate_last = 0.0  # tokens/s from the last prefill chunk cadence
        self._pf_last: tuple[float, int] | None = None
        self.spec_accepted = 0
        self.spec_proposed = 0
        self.moe: dict | None = None

    @property
    def active(self) -> int:
        return len(self._inflight)

    @property
    def inflight_uids(self) -> tuple[int, ...]:
        """Stable snapshot used by prepare-stop to abort every still-admitted request."""
        return tuple(sorted(self._inflight))

    def on_new_user(self, uid: int) -> None:
        if not self._inflight:
            # Starting from idle: drop the previous run's residual pool gauges.
            self.kv_used_pages = 0
            self.mamba_used_slots = 0
            self.swa_used_tokens = 0
            self.prefill_active = False
            self.prompt_processed = 0
            self.prompt_total = 0
            self.queue_reqs = 0
        self._inflight.add(uid)
        self._aborting.discard(uid)

    def on_abort(self, uid: int) -> None:
        if uid in self._inflight:
            self._aborting.add(uid)

    def observe(self, reply: Any, now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        if getattr(reply, "completion_tokens_delta", 0) > 0:
            self._decode.append((t, reply.completion_tokens_delta))
            self.completion_tokens_total += reply.completion_tokens_delta
        if getattr(reply, "prompt_tokens_delta", 0) > 0:
            self._prefill.append((t, reply.prompt_tokens_delta))
            self.prompt_tokens_total += reply.prompt_tokens_delta
        self.cached_tokens_total += getattr(reply, "cached_tokens", 0) or 0
        # Terminal handling first, so the pool gauges below see the post-finish flight count.
        if getattr(reply, "finished", False):
            uid = getattr(reply, "uid", None)
            if uid in self._inflight:
                self._inflight.discard(uid)
                if uid in self._aborting:
                    self._aborting.discard(uid)
                else:
                    self.completed += 1
        # Pool gauges: a single batch's snapshot can read 0 while another request still
        # holds the pool, so keep the peak while anything is in flight and drop to 0 the
        # moment the engine goes idle.
        inflight = self.active
        if getattr(reply, "kv_total_pages", 0) > 0:  # ignore 0/0 (prompt reply, owned-KV)
            self.kv_total_pages = reply.kv_total_pages
            self.kv_used_pages = max(self.kv_used_pages, reply.kv_used_pages) if inflight else 0
            if getattr(reply, "kv_page_size", 0) > 0:
                self.kv_page_size = reply.kv_page_size
        if getattr(reply, "mamba_total_slots", 0) > 0:  # hybrid (GDN) only
            self.mamba_total_slots = reply.mamba_total_slots
            self.mamba_used_slots = max(self.mamba_used_slots, reply.mamba_used_slots) if inflight else 0
        if getattr(reply, "swa_total_tokens", 0) > 0:  # SWA (window pool) only
            self.swa_total_tokens = reply.swa_total_tokens
            self.swa_used_tokens = max(self.swa_used_tokens, reply.swa_used_tokens) if inflight else 0
        if getattr(reply, "gpu_mem_bytes", 0) > 0:
            self.vram_bytes = reply.gpu_mem_bytes
        self.queue_reqs = getattr(reply, "queue_reqs", self.queue_reqs) or 0
        if getattr(reply, "prefill_active", False):
            self.prefill_active = True
            self.prompt_processed = reply.prompt_processed
            self.prompt_total = reply.prompt_total
            if self._pf_last is not None and t > self._pf_last[0]:
                d = reply.prompt_processed - self._pf_last[1]
                if d > 0:
                    self.prefill_rate_last = d / (t - self._pf_last[0])
            self._pf_last = (t, reply.prompt_processed)
        elif getattr(reply, "completion_tokens_delta", 0) > 0:
            self.prefill_active = False  # decoding now
            self._pf_last = None
        if getattr(reply, "spec_proposed", 0) > 0:
            self.spec_accepted = reply.spec_accepted
            self.spec_proposed = reply.spec_proposed
        if getattr(reply, "moe", None):
            self.moe = reply.moe

    def _rate(self, window: "deque[tuple[float, int]]", now: float | None) -> float:
        t = time.monotonic() if now is None else now
        cutoff = t - self.window_s
        while window and window[0][0] < cutoff:
            window.popleft()
        if not window:
            return 0.0
        total = sum(n for _ts, n in window)
        span = max(t - window[0][0], 1e-9)
        return total / span

    def decode_tps(self, now: float | None = None) -> float:
        return self._rate(self._decode, now)

    def prefill_tps(self, now: float | None = None) -> float:
        return self._rate(self._prefill, now)


def _sum_or_none(docs: list, getter) -> Any:
    vals = [getter(d) for d in docs]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def build_dp_stats(dp: Any, p95_ms: int, ttft_mean_ms: int) -> dict:
    """Aggregate /v1/stats for a data-parallel serve: the SAME top-level shape as build_stats
    but totalled across engines, plus ``data_parallel`` {size, routing} and ``engines`` (one
    compact block per engine). Counters/tokens/throughput are summed (engines run in
    parallel, so summed decode_tps is the real serve throughput); pool gauges are summed;
    model/uptime/instance come from the aggregate."""
    engines = list(getattr(dp, "engines", None) or [])
    docs = [build_stats(e.frontend, p95_ms, ttft_mean_ms) for e in engines]
    base = dict(docs[0]) if docs else {}
    ready = [e.ready_at for e in engines if getattr(e, "ready_at", None) is not None]
    fully = bool(ready) and len(ready) == len(engines)
    now = time.monotonic()
    base["uptime_s"] = max(0, int(now - max(ready))) if fully else 0  # type: ignore[arg-type]
    base["instance_id"] = getattr(dp, "instance_id", None)
    base["gpus"] = [g for d in docs for g in (d.get("gpus") or [])]

    def rq(d, k):
        return (d.get("requests") or {}).get(k)

    base["requests"] = {
        "active": _sum_or_none(docs, lambda d: rq(d, "active")) or 0,
        "completed": _sum_or_none(docs, lambda d: rq(d, "completed")) or 0,
        "queued": _sum_or_none(docs, lambda d: rq(d, "queued")) or 0,
        "p95_ms": p95_ms,
        "ttft_mean_ms": ttft_mean_ms,
        "prompt_tokens_total": _sum_or_none(docs, lambda d: rq(d, "prompt_tokens_total")) or 0,
        "cached_tokens_total": _sum_or_none(docs, lambda d: rq(d, "cached_tokens_total")) or 0,
        "completion_tokens_total": _sum_or_none(docs, lambda d: rq(d, "completion_tokens_total")) or 0,
    }
    base["throughput"] = {
        "decode_tps": round(_sum_or_none(docs, lambda d: (d.get("throughput") or {}).get("decode_tps")) or 0.0, 1),
        "prefill_tps": round(_sum_or_none(docs, lambda d: (d.get("throughput") or {}).get("prefill_tps")) or 0.0, 1),
    }
    base["vram_bytes"] = _sum_or_none(docs, lambda d: d.get("vram_bytes")) or 0

    kv0 = base.get("kv") or {}
    if any((d.get("kv") or {}).get("total_pages") for d in docs):
        base["kv"] = {
            "used_pages": _sum_or_none(docs, lambda d: (d.get("kv") or {}).get("used_pages")) or 0,
            "total_pages": _sum_or_none(docs, lambda d: (d.get("kv") or {}).get("total_pages")) or 0,
            "page_size": kv0.get("page_size", 1),
            "quant": kv0.get("quant", "none"),
        }
    else:
        base["kv"] = None
    if any((d.get("mamba") or {}).get("total_slots") for d in docs):
        base["mamba"] = {
            "used_slots": _sum_or_none(docs, lambda d: (d.get("mamba") or {}).get("used_slots")) or 0,
            "total_slots": _sum_or_none(docs, lambda d: (d.get("mamba") or {}).get("total_slots")) or 0,
        }
    if any((d.get("swa") or {}).get("total_pages") for d in docs):
        base["swa"] = {
            "used_pages": _sum_or_none(docs, lambda d: (d.get("swa") or {}).get("used_pages")) or 0,
            "total_pages": _sum_or_none(docs, lambda d: (d.get("swa") or {}).get("total_pages")) or 0,
            "page_size": (base.get("swa") or {}).get("page_size", 1),
        }
    base["prefill"] = next((d.get("prefill") for d in docs if d.get("prefill")), None)
    if any(d.get("spec") for d in docs):
        base["spec"] = {
            "accepted": _sum_or_none(docs, lambda d: (d.get("spec") or {}).get("accepted")) or 0,
            "proposed": _sum_or_none(docs, lambda d: (d.get("spec") or {}).get("proposed")) or 0,
            "rate": None,
        }
    base["moe"] = next((d.get("moe") for d in docs if d.get("moe")), None)

    base["engines"] = [
        {
            "index": e.index,
            "state": e.maintenance_state,
            "gpu": e.gpu(),
            "in_flight": e.in_flight(),
            "restarts": getattr(e, "restarts", 0),
            **{
                k: d.get(k) for k in ("requests", "throughput", "kv", "mamba", "swa",
                                       "vram_bytes", "prefill", "spec", "moe")
            },
        }
        for e, d in zip(engines, docs)
    ]
    base["data_parallel"] = {
        "size": len(engines),
        "serving": sum(1 for e in engines if e.ready),
        "routing": dp.routing_stats() if hasattr(dp, "routing_stats") else {},
    }
    return base


def derive_model_card(config: Any) -> dict:
    """attn enum + moe bool + ctx from the model config; input_modalities is what the API accepts right now."""
    mc = config.model_config
    if getattr(mc, "has_linear_attention", False):
        attn = "hybrid_linear"
    elif getattr(mc, "has_swa_attention", False):
        attn = "hybrid_swa"
    else:
        attn = "mha"
    return {
        "id": config.served_model_name,
        "ctx": config.max_seq_len,
        "attn": attn,
        "moe": bool(getattr(mc, "is_moe", False)),
        "input_modalities": ["text", *sorted(getattr(config, "served_modalities", ()))],
    }


def _swa_page_size(config: Any) -> int:
    """The window pool's own page unit: P (window_size) for DSV4, 1 token for radix-SWA.
    Mirrors compute_cache_pools' swa_page_size."""
    dsv4 = getattr(getattr(config, "model_config", None), "dsv4_args", None)
    if dsv4 is not None:
        return int(getattr(dsv4, "window_size", 0) or 1)
    return 1


def build_stats(state: Any, p95_ms: int, ttft_mean_ms: int) -> dict:
    """Full /v1/stats doc. throughput is 0 when idle; kv/mamba/swa are null
    when their total is 0 (owned-KV / non-hybrid / non-SWA). kv and swa share one shape:
    pages + the pool's own page_size (tokens = pages x page_size). gpus: the engine's GPU as
    [{index, name, uuid, total_bytes}] (the primary rank's; a list so TP can extend it), []
    until the readiness meta arrives."""
    tr: StatsTracker = state.stats
    config = state.config
    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    kv = (
        {"used_pages": tr.kv_used_pages, "total_pages": tr.kv_total_pages,
         # Prefer the engine's resolved page size; the CLI value is what the parent
         # itself parsed and can be stale (a backend override is applied in-engine).
         "page_size": tr.kv_page_size or getattr(config, "page_size", 1),
         "quant": getattr(config, "kv_quant", "none")}
        if tr.kv_total_pages > 0 else None
    )
    mamba = (
        {"used_slots": tr.mamba_used_slots, "total_slots": tr.mamba_total_slots}
        if tr.mamba_total_slots > 0 else None
    )
    sps = _swa_page_size(config)
    swa = (
        {"used_pages": tr.swa_used_tokens // sps, "total_pages": tr.swa_total_tokens // sps,
         "page_size": sps}
        if tr.swa_total_tokens > 0 else None
    )
    prefill = None
    if tr.prefill_active and tr.prompt_total > 0:
        eta = None
        rate = tr.prefill_rate_last or tr.prefill_tps()
        if rate > 0:
            eta = round(max(0, tr.prompt_total - tr.prompt_processed) / rate, 2)
        prefill = {
            "processed_tokens": tr.prompt_processed,
            "total_tokens": tr.prompt_total,
            "usage_ratio": tr.prompt_processed / tr.prompt_total,
            "eta_seconds": eta,
        }
    spec = None
    if tr.spec_proposed > 0:
        spec = {"accepted": tr.spec_accepted, "proposed": tr.spec_proposed,
                "rate": tr.spec_accepted / tr.spec_proposed}
    moe = None
    if tr.moe:
        moe = {
            "miss_ratio": tr.moe.get("miss_rate"),
            "cache_size": tr.moe.get("slots_per_layer") or tr.moe.get("cache_size"),
            "hybrid_fetch_fraction": tr.moe.get("hybrid_fetch_fraction"),
            "oracle_hit": tr.moe.get("oracle_hit_at_slots"),
        }
    return {
        "instance_id": getattr(state, "instance_id", None),
        "model": derive_model_card(config),
        "uptime_s": uptime_s,
        "kv": kv,
        "mamba": mamba,
        "swa": swa,
        "vram_bytes": tr.vram_bytes,
        "gpus": list(getattr(state, "gpus", None) or []),
        "throughput": {
            "decode_tps": round(tr.decode_tps(), 1),
            "prefill_tps": round(tr.prefill_tps(), 1),
        },
        "requests": {
            "active": tr.active,
            "completed": tr.completed,
            "queued": tr.queue_reqs,
            "p95_ms": p95_ms,
            "ttft_mean_ms": ttft_mean_ms,
            "prompt_tokens_total": tr.prompt_tokens_total,
            "cached_tokens_total": tr.cached_tokens_total,
            "completion_tokens_total": tr.completion_tokens_total,
        },
        "prefill": prefill,
        "spec": spec,
        "moe": moe,
    }
