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
from fastapi.responses import JSONResponse
from freetoken.core import SamplingParams
from freetoken.message import TokenizeMsg

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
) -> None:
    @app.get("/health")
    async def health():
        return build_health(get_state(), app.version)

    from . import request_ring

    @app.get("/v1/requests")
    async def list_requests(since: int = 0, limit: int = 100):
        limit = max(1, min(limit, 512))
        entries, next_cursor = request_ring.requests_since(since, limit)
        return {"entries": entries, "next_cursor": next_cursor}

    from .stats import build_stats

    @app.get("/v1/stats")
    async def stats():
        doc = build_stats(
            get_state(), request_ring.requests_p95_ms(), request_ring.requests_ttft_mean_ms()
        )
        # Surface the model's recommended sampling (from its generation_config.json / GGUF
        # metadata) so clients can seed their sampling controls per-model instead of guessing.
        if get_model_sampling is not None:
            doc["model"]["sampling"] = get_model_sampling() or {}
        return doc

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
