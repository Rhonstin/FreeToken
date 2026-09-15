"""In-engine data-parallel (DP) support: N independent engines behind one serve process.

A DP serve spawns one single-GPU engine per GPU (each with its own scheduler, KV cache,
radix cache and request queue) and routes every request to one of them here, inside the
engine process — there is no external load balancer. All engines share the host-RAM expert
pool via FREETOKEN_SHARED_BANKS (env is inherited by the children).

This module is deliberately dependency-free with respect to api_server (the concrete
``FrontendManager`` is passed in as an opaque handle) so it stays unit-testable.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, List


def _field(hint: Any, name: str) -> Any:
    """Read ``name`` off a request hint that may be a pydantic model, a dict, or a namespace."""
    if isinstance(hint, dict):
        return hint.get(name)
    return getattr(hint, name, None)


def sticky_key(hint: Any) -> str | None:
    """Conversation key for engine affinity, or None when the request has no usable signal.

    A conversation id (any of the usual spellings) wins; otherwise a hash of the leading
    prompt/messages — the strongest radix-cache signal. Bounded slice so a giant prompt never
    costs more than one sha1 over ~8 KiB."""
    if hint is None:
        return None
    if isinstance(hint, str):
        return hashlib.sha1(hint[:8192].encode("utf-8", "ignore")).hexdigest()
    for attr in ("conversation_id", "conversationId", "session_id", "sessionId"):
        v = _field(hint, attr)
        if isinstance(v, str) and v:
            return "cid:" + v
    user = _field(hint, "user")
    if isinstance(user, str) and user:
        return "user:" + user
    messages = _field(hint, "messages")
    if messages:
        return "sha1:" + hashlib.sha1(repr(messages)[:8192].encode("utf-8", "ignore")).hexdigest()
    prompt = _field(hint, "prompt")
    if isinstance(prompt, str) and prompt:
        return "sha1:" + hashlib.sha1(prompt[:8192].encode("utf-8", "ignore")).hexdigest()
    return None


def restart_delay(consecutive_failures: int, base_s: float = 1.0, cap_s: float = 30.0) -> float:
    """Exponential backoff for a DP engine group restart, capped. Attempt n (1-based) waits
    base * 2**(n-1), so the first restart is quick and a crash loop backs off."""
    n = max(1, int(consecutive_failures))
    return min(cap_s, base_s * (2 ** (n - 1)))


@dataclass
class EngineRuntime:
    """One DP engine: its FrontendManager plus the lifecycle bookkeeping the DP server needs."""

    index: int
    config: Any
    frontend: Any
    progress: Any = None
    handle: Any = None
    ready_at: float | None = None
    # Watchdog bookkeeping.
    spawn: Any = None  # callable -> new BackendHandle for this engine (group restart)
    restarts: int = 0
    consecutive_failures: int = 0
    last_failure: str | None = None
    last_failure_at: float | None = None
    last_ready_at: float | None = None

    def in_flight(self) -> int:
        try:
            return len(getattr(self.frontend, "ack_map", None) or {})
        except Exception:  # noqa: BLE001 -- a wedged manager must not break routing
            return 0

    @property
    def maintenance_state(self) -> str:
        return getattr(self.frontend, "maintenance_state", "serving")

    @property
    def ready(self) -> bool:
        return self.maintenance_state == "serving"

    def gpu(self) -> dict:
        """This engine's GPU: prefer the readiness meta ({index,name,uuid,total_bytes}), fall
        back to the configured --gpu entry so /health is never blank on a fresh engine."""
        gpus = getattr(self.frontend, "gpus", None) or []
        if gpus:
            g = dict(gpus[0])
            g.setdefault("rank", 0)
            return g
        assigned = getattr(self.config, "gpu_assigned", None) or ()
        raw = getattr(self.config, "gpu", None) or ()
        entry = (assigned or raw or (None,))[0]
        return {"index": None, "name": None, "uuid": entry, "total_bytes": 0, "rank": 0}

    def as_health(self) -> dict:
        st = getattr(self.frontend, "stats", None)
        return {
            "index": self.index,
            "state": self.maintenance_state,
            "in_flight": self.in_flight(),
            "queued": int(getattr(st, "queue_reqs", 0) or 0) if st is not None else 0,
            "completed": int(getattr(st, "completed", 0) or 0) if st is not None else 0,
            "gpu": self.gpu(),
            "expected_acks": int(getattr(self.handle, "expected_acks", 0) or 0),
            "restarts": self.restarts,
            "consecutive_failures": self.consecutive_failures,
            "last_failure": self.last_failure,
            "last_failure_age_s": (
                round(time.monotonic() - self.last_failure_at, 3) if self.last_failure_at else None
            ),
            "ready_at_age_s": (
                round(time.monotonic() - self.last_ready_at, 3) if self.last_ready_at else None
            ),
            "fatal_error": getattr(self.frontend, "fatal_error", None),
        }


class AggProgress:
    """Sum of the per-engine load progress: /health wants one percentage for the whole DP
    serve. ``phase`` is the phase of the first not-yet-ready engine (else "warmup")."""

    def __init__(self, engines: List[EngineRuntime]) -> None:
        self._engines = engines

    def _progress_of(self, e: EngineRuntime):
        return getattr(e.frontend, "load_progress", None) or e.progress

    @property
    def done_bytes(self) -> int:
        return sum(int(getattr(self._progress_of(e), "done_bytes", 0) or 0) for e in self._engines)

    @property
    def total_bytes(self) -> int:
        return sum(int(getattr(self._progress_of(e), "total_bytes", 0) or 0) for e in self._engines)

    @property
    def phase(self) -> str:
        for e in self._engines:
            if not e.ready:
                lp = self._progress_of(e)
                ph = getattr(lp, "phase", None)
                if ph and ph != "other":
                    return ph
        return "warmup"


class DPServer:
    """Owns the N engines and the routing policy; presents an aggregate lifecycle surface
    (maintenance_state / load_progress / ready_at / backend_processes) so the existing
    health/control/accounting code can read it like a single FrontendManager."""

    def __init__(self, config: Any, engines: List[EngineRuntime]) -> None:
        self.config = config
        self.engines = engines
        self._sticky: dict[str, int] = {}
        self._rr: int = 0
        self._lock = threading.Lock()
        self.routed: dict[int, int] = {e.index: 0 for e in engines}
        self.sticky_hits: int = 0
        self.sticky_misses: int = 0

    def __getattr__(self, name: str) -> Any:
        """Fall through to engine 0 for per-engine state the aggregate has no opinion on
        (stats, last_rebuild, unit_bytes, cache_pools, free_vram_bytes, ...). Keeps the
        existing single-engine read paths working unchanged in DP mode; the cache-geometry
        control plane therefore reflects engine 0."""
        try:
            engines = object.__getattribute__(self, "engines")
        except AttributeError:
            raise AttributeError(name) from None
        if engines:
            return getattr(engines[0].frontend, name)
        raise AttributeError(name)

    # -- routing -----------------------------------------------------------------
    def route(self, hint: Any = None):
        candidates = [e for e in self.engines if e.ready]
        if not candidates:
            return self.engines[0].frontend
        key = sticky_key(hint)
        if key is not None:
            with self._lock:
                idx = self._sticky.get(key)
            if idx is not None:
                for e in candidates:
                    if e.index == idx:
                        with self._lock:
                            self.sticky_hits += 1
                            self.routed[e.index] = self.routed.get(e.index, 0) + 1
                        return e.frontend
        # Least in-flight, with a round-robin tie-break. The tie-break matters: two requests
        # that arrive together both see every engine idle (a request's ack_map entry is only
        # created once it reaches new_user, after async pre-render), so a pure min would pin
        # both to engine 0. Rotating among the tied minima spreads them instead.
        best = min(e.in_flight() for e in candidates)
        tied = [e for e in candidates if e.in_flight() == best]
        with self._lock:
            chosen = tied[self._rr % len(tied)]
            self._rr += 1
            self.routed[chosen.index] = self.routed.get(chosen.index, 0) + 1
            if key is not None:
                self.sticky_misses += 1
                if len(self._sticky) >= 4096:
                    self._sticky.clear()
                self._sticky[key] = chosen.index
        return chosen.frontend

    def routing_stats(self) -> dict:
        with self._lock:
            return {
                "routed": dict(self.routed),
                "sticky_keys": len(self._sticky),
                "sticky_hits": self.sticky_hits,
                "sticky_misses": self.sticky_misses,
                "round_robin": self._rr,
            }

    def engine_of(self, frontend: Any) -> EngineRuntime | None:
        for e in self.engines:
            if e.frontend is frontend:
                return e
        return None

    # -- aggregate lifecycle -----------------------------------------------------
    @property
    def maintenance_state(self) -> str:
        states = [e.maintenance_state for e in self.engines]
        if not states:
            return "loading"
        if any(s == "serving" for s in states):
            return "serving"
        for s in ("loading", "rebuilding", "stopping", "failed"):
            if any(x == s for x in states):
                return s
        return states[0]

    @property
    def fatal_error(self) -> str | None:
        # Whole-serve fatality only when EVERY engine is dead; one dead engine is a partial
        # outage the other engines keep covering (that is the point of DP).
        if self.engines and all(e.maintenance_state == "failed" for e in self.engines):
            return getattr(self.engines[0].frontend, "fatal_error", None) or "all DP engines failed"
        return None

    @property
    def instance_id(self) -> str | None:
        return getattr(self.engines[0].frontend, "instance_id", None) if self.engines else None

    @property
    def load_progress(self) -> AggProgress:
        return AggProgress(self.engines)

    @property
    def ready_at(self) -> float | None:
        snaps = [e.ready_at for e in self.engines]
        if snaps and all(s is not None for s in snaps):
            return max(snaps)  # type: ignore[type-var]
        return None

    @property
    def stats(self) -> Any:
        return getattr(self.engines[0].frontend, "stats", None) if self.engines else None

    @property
    def gpus(self) -> list:
        out: list = []
        for e in self.engines:
            out.extend(getattr(e.frontend, "gpus", None) or [])
        return out

    @property
    def backend_processes(self) -> list:
        procs: list = []
        for e in self.engines:
            procs.extend(getattr(e.frontend, "backend_processes", None) or [])
        return procs

    def engines_health(self) -> list:
        return [e.as_health() for e in self.engines]

    def fail_pending_rebuilds(self, message: str) -> None:
        for e in self.engines:
            try:
                e.frontend.fail_pending_rebuilds(message)
            except Exception:  # noqa: BLE001 -- best-effort on a failing engine
                pass

    def shutdown(self) -> None:
        for e in self.engines:
            try:
                e.frontend.shutdown()
            except Exception:  # noqa: BLE001 -- tear down what we can
                pass
