"""Control-plane endpoints consumed by the desktop app and internal tooling: /health
(lifecycle), /v1/stats (runtime metrics, Task 6), /v1/requests (request log ring, Task 5),
and /v1/score (teacher-forced NLL/PPL over raw text, for KV-cache quality measurement).

All read-only handlers read a shared FrontendManager snapshot via ``get_state``; nothing
there touches the scheduler or blocks. /v1/score is the one exception: it submits a scoring
request through the normal tokenizer/scheduler plumbing and awaits its chunked replies.
Registered on the app alongside the OpenAI/Anthropic/Responses routes.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg

from . import metrics
from .api_models import ScoreRequest


def build_health(state: Any, version: str) -> dict:
    """Full-lifecycle health doc: loading -> ok -> error."""
    instance_id = getattr(state, "instance_id", None)
    fatal = getattr(state, "fatal_error", None)
    if fatal:
        return {"status": "error", "message": fatal, "instance_id": instance_id}

    mstate = getattr(state, "maintenance_state", "serving")
    config = getattr(state, "config", None)
    model = getattr(config, "served_model_name", None)

    if mstate == "loading":
        lp = getattr(state, "load_progress", None)
        return {
            "status": "loading",
            "phase": lp.phase if lp is not None else "other",
            "progress": {
                "done_bytes": lp.done_bytes if lp is not None else 0,
                "total_bytes": lp.total_bytes if lp is not None else 0,
            },
            "model": model,
            "instance_id": instance_id,
        }

    ready_at = getattr(state, "ready_at", None)
    uptime_s = max(0, int(time.monotonic() - ready_at)) if ready_at is not None else 0
    return {
        "status": "ok",
        "model": model,
        "instance_id": instance_id,
        "uptime_s": uptime_s,
        "maintenance": mstate,
        "version": version,
    }


def register_control_routes(
    app: FastAPI,
    get_state: Callable[[], Any],
    get_model_sampling: Callable[[], dict] | None = None,
    get_dp_state: Callable[[], Any] | None = None,
) -> None:
    # health/stats describe the WHOLE serve; in DP that is the aggregate DPServer, not one
    # engine. get_state still names the (routed) engine for /v1/score and cache control.
    _agg = get_dp_state or get_state

    @app.get("/health")
    async def health():
        dp = _agg()
        doc = build_health(dp, app.version)
        engines = getattr(dp, "engines_health", None)
        if callable(engines):
            rows = engines()
            doc["engines"] = rows
            doc["data_parallel"] = {
                "size": len(rows),
                "serving": sum(1 for e in rows if e.get("state") == "serving"),
            }
        return doc

    from . import request_ring

    @app.get("/v1/requests")
    async def list_requests(since: int = 0, limit: int = 100):
        limit = max(1, min(limit, 512))
        entries, next_cursor = request_ring.requests_since(since, limit)
        return {"entries": entries, "next_cursor": next_cursor}

    from .stats import build_dp_stats, build_stats

    def _stats_doc() -> dict:
        """The shared /v1/stats document: engine snapshot + ring percentiles + live GPU.
        In DP the snapshot is aggregated across every engine (build_dp_stats)."""
        from . import request_ring

        state = _agg()
        p95 = request_ring.requests_p95_ms()
        ttft = request_ring.requests_ttft_mean_ms()
        if hasattr(state, "engines"):
            doc = build_dp_stats(state, p95, ttft)
        else:
            doc = build_stats(state, p95, ttft)
        doc.setdefault("requests", {}).update(request_ring.requests_latency())
        # Surface the model's recommended sampling (from its generation_config.json / GGUF
        # metadata) so clients can seed their sampling controls per-model instead of guessing.
        if get_model_sampling is not None:
            doc.setdefault("model", {})["sampling"] = get_model_sampling() or {}
        doc["gpu_live"] = metrics.gpu_snapshot()
        return doc

    @app.get("/v1/stats")
    async def stats():
        return _stats_doc()

    @app.get("/v1/metrics")
    async def metrics_json():
        """JSON view of the same document /metrics renders as Prometheus text."""
        return _stats_doc()

    @app.get("/metrics")
    async def metrics_prometheus():
        return Response(content=metrics.to_prometheus(_stats_doc()),
                        media_type="text/plain; version=0.0.4; charset=utf-8")

    def _arena_status() -> dict:
        """Shared host expert-pool (tmpfs arena) state, from the env pointer the engine used."""
        import os

        path = os.environ.get("FREETOKEN_SHARED_BANKS")
        if not path:
            return {"enabled": False}
        out: dict = {"enabled": True, "path": path}
        for suffix, key in (("", "file"), (".ready", "ready"), (".lock", "lock")):
            p = path + suffix
            try:
                st = os.stat(p)
                out[key] = {
                    "exists": True,
                    "size_bytes": st.st_size,
                    "age_s": round(time.time() - st.st_mtime, 1),
                }
            except OSError:
                out[key] = {"exists": False}
        return out

    @app.get("/v1/dp/status")
    async def dp_status():
        """Detailed DP monitoring: per-engine state/queues/throughput/VRAM + routing counters +
        shared-arena state + live GPU telemetry. Works for a single-engine serve too (size 1)."""
        doc = _stats_doc()
        return {
            "mode": "data_parallel" if (doc.get("data_parallel") or {}).get("size", 1) > 1 else "single",
            "instance_id": doc.get("instance_id"),
            "model": doc.get("model"),
            "uptime_s": doc.get("uptime_s"),
            "data_parallel": doc.get("data_parallel"),
            "engines": doc.get("engines"),
            "totals": {
                "requests": doc.get("requests"),
                "throughput": doc.get("throughput"),
                "vram_bytes": doc.get("vram_bytes"),
                "kv": doc.get("kv"),
                "mamba": doc.get("mamba"),
                "swa": doc.get("swa"),
                "prefill": doc.get("prefill"),
                "spec": doc.get("spec"),
                "moe": doc.get("moe"),
            },
            "arena": _arena_status(),
            "gpu_live": metrics.gpu_snapshot(),
        }

    @app.post("/v1/score")
    async def score(req: ScoreRequest):
        """Teacher-forced scoring of raw text: ``nll[i] = -log p(token_{i+1} | tokens_{<=i})``.

        No chat template is applied (a PPL corpus is scored as-is, matching a raw
        /v1/completions prompt). The response's ``n_tokens`` is the prompt token count and
        ``nll`` holds ``n_tokens - 1`` values (the final token has no successor); ``ppl`` is
        ``exp(mean(nll))``, None when the text tokenizes to a single token. The scheduler
        forwards the text in bounded chunks and streams one NLL list per chunk, accumulated
        here; ``top1_rate`` is the fraction of positions whose argmax equals the corpus token.
        """
        if not req.text:
            return JSONResponse(
                {
                    "error": {
                        "message": "text must not be empty",
                        "type": "invalid_request_error",
                        "code": None,
                    }
                },
                status_code=400,
            )
        state = get_state()
        # Raises AdmissionClosedError (503 via the app exception handler) when the engine
        # is loading/rebuilding/stopping/failed.
        uid = state.new_user()
        await state.send_one(
            TokenizeMsg(
                uid=uid,
                text=req.text,
                sampling_params=SamplingParams(max_tokens=0),
                score_only=True,
                score_chunk=req.chunk,
            )
        )
        nlls: list[float] = []
        top1_hits = 0
        n_tokens = 0
        async for ack in state.wait_for_ack(uid):
            if ack.error:
                # The scheduler classifies a prompt longer than the servable context as
                # context_length_exceeded; every other pre-admission failure is a 400
                # (e.g. text that tokenizes to zero tokens).
                status = 413 if ack.error_code == "context_length_exceeded" else 400
                return JSONResponse(
                    {
                        "error": {
                            "message": ack.error,
                            "type": "invalid_request_error",
                            "code": ack.error_code,
                        }
                    },
                    status_code=status,
                )
            n_tokens += ack.prompt_tokens_delta
            if ack.nlls:
                nlls.extend(ack.nlls)
            top1_hits += ack.top1_hits
            if ack.finished:
                break
        if not nlls:
            n_tokens = n_tokens or 1
        if nlls and not all(math.isfinite(x) for x in nlls):
            return JSONResponse(
                {
                    "error": {
                        "message": "scoring produced a non-finite NLL",
                        "type": "server_error",
                        "code": None,
                    }
                },
                status_code=500,
            )
        return {
            "nll": nlls,
            "ppl": math.exp(sum(nlls) / len(nlls)) if nlls else None,
            "n_tokens": n_tokens,
            "top1_hits": top1_hits,
            "top1_rate": top1_hits / len(nlls) if nlls else 0.0,
        }
