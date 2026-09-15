"""DP control-plane: aggregated /v1/stats, aggregate /health, per-engine metrics, and DP-wide
prepare-stop. Pure fakes — no CUDA, no ZMQ, no worker processes."""

from __future__ import annotations

import asyncio
import types

from freetoken.server.accounting import prepare_stop_all
from freetoken.server.dp import DPServer, EngineRuntime
from freetoken.server.metrics import to_prometheus
from freetoken.server.stats import StatsTracker, build_dp_stats


def _config(name="m"):
    mc = types.SimpleNamespace(
        has_linear_attention=False, has_swa_attention=False, is_moe=True, num_experts=8, num_moe_layers=2
    )
    return types.SimpleNamespace(
        served_model_name=name, max_seq_len=4096, page_size=1, kv_quant="none",
        model_config=mc, moe_cache_policy="lru",
    )


class FakeFM:
    def __init__(self, name="m") -> None:
        self.config = _config(name)
        self.stats = StatsTracker()
        self.maintenance_state = "serving"
        self.ack_map: dict = {}
        self.event_map: dict = {}
        self.gpus = []
        self.instance_id = f"inst-{name}"
        self.ready_at = 1000.0
        self.fatal_error = None
        self.last_rebuild = None
        self.rebuild_futures: dict = {}
        self._accounting_prepare_lock = None

    async def abort_user(self, uid):  # pragma: no cover - only hit on drain timeout
        self.stats.on_abort(uid)


def _engine(i, name="m", *, active=0, completed=0, decode=0):
    fm = FakeFM(name)
    fm.stats.completed = completed
    fm.stats.prompt_tokens_total = 100 * (i + 1)
    fm.stats.completion_tokens_total = 50 * (i + 1)
    for u in range(active):
        fm.stats._inflight.add(1000 + i * 10 + u)
    fm.gpus = [{"index": i, "name": "RTX 3090", "uuid": f"GPU-{i}", "total_bytes": 24}]
    fm.ready_at = 1000.0 + i
    rt = EngineRuntime(index=i, config=fm.config, frontend=fm)
    rt.ready_at = fm.ready_at
    return rt


def test_build_dp_stats_aggregates() -> None:
    dp = DPServer(config=_config(), engines=[_engine(0, active=1, completed=3), _engine(1, active=2, completed=4)])
    doc = build_dp_stats(dp, p95_ms=12, ttft_mean_ms=5)
    assert doc["requests"]["active"] == 3
    assert doc["requests"]["completed"] == 7
    assert doc["requests"]["prompt_tokens_total"] == 100 + 200
    assert doc["requests"]["completion_tokens_total"] == 50 + 100
    assert doc["data_parallel"]["size"] == 2 and doc["data_parallel"]["serving"] == 2
    assert len(doc["engines"]) == 2
    assert len(doc["gpus"]) == 2
    assert doc["requests"]["p95_ms"] == 12


def test_prometheus_has_per_engine_series() -> None:
    dp = DPServer(config=_config(), engines=[_engine(0, completed=1), _engine(1, completed=2)])
    doc = build_dp_stats(dp, p95_ms=0, ttft_mean_ms=0)
    text = to_prometheus(doc)
    assert 'freetoken_dp_engines 2' in text
    assert 'freetoken_engine_up{engine="0"}' in text
    assert 'freetoken_engine_completed_total{engine="1"} 2' in text


def test_prepare_stop_all_seals_every_engine() -> None:
    dp = DPServer(config=_config(), engines=[_engine(0), _engine(1)])

    async def _run():
        # same loop for both calls: the per-engine/dp locks are asyncio primitives
        return (
            await prepare_stop_all(dp, drain_timeout_s=0.1, abort_timeout_s=0.1),
            await prepare_stop_all(dp, drain_timeout_s=0.1, abort_timeout_s=0.1),
        )

    result, again = asyncio.run(_run())
    assert result["drain_complete"] is True
    assert result["engines"] == 2
    assert result["prompt_tokens_total"] == 100 + 200
    assert all(e.frontend.maintenance_state == "stopping" for e in dp.engines)
    assert dp.maintenance_state == "stopping"
    assert again == result  # idempotent: cached snapshot on retry


def test_health_doc_aggregates_dp() -> None:
    from freetoken.server.control_api import build_health

    dp = DPServer(config=_config(), engines=[_engine(0), _engine(1)])
    doc = build_health(dp, "0.0.0-test")
    assert doc["status"] == "ok"
    assert doc["maintenance"] == "serving"
