"""DP (data-parallel) unit tests: routing policy, sticky affinity, address isolation, and the
argument-level guards. No CUDA, no worker processes — Fakes stand in for FrontendManager."""

from __future__ import annotations

import dataclasses

import pytest

from freetoken.server.dp import DPServer, EngineRuntime, sticky_key


class FakeFM:
    def __init__(self, state: str = "serving") -> None:
        self.maintenance_state = state
        self.ack_map: dict = {}
        self.stats = None
        self.gpus: list = []
        self.backend_processes: list = []
        self.fatal_error = None
        self.instance_id = f"inst-{id(self)}"
        self.ready_at = None
        self.load_progress = None
        self.failed_msg = None
        self.shut = False

    def fail_pending_rebuilds(self, message: str) -> None:
        self.failed_msg = message

    def shutdown(self) -> None:
        self.shut = True


def mk(index: int, state: str = "serving", inflight: int = 0) -> EngineRuntime:
    fm = FakeFM(state)
    for k in range(inflight):
        fm.ack_map[k] = []
    return EngineRuntime(index=index, config=object(), frontend=fm)


def test_least_in_flight() -> None:
    dp = DPServer(config=object(), engines=[mk(0, inflight=3), mk(1, inflight=1)])
    assert dp.route() is dp.engines[1].frontend


def test_sticky_keeps_conversation_on_one_engine() -> None:
    a, b = mk(0), mk(1)
    dp = DPServer(config=object(), engines=[a, b])
    first = dp.route({"messages": [{"role": "user", "content": "hello world"}]})
    # make the first engine look busier; stickiness must still win
    a.frontend.ack_map[0] = []
    a.frontend.ack_map[1] = []
    again = dp.route({"messages": [{"role": "user", "content": "hello world"}]})
    assert again is first


def test_failed_engine_is_skipped() -> None:
    good, bad = mk(0, state="failed"), mk(1)
    dp = DPServer(config=object(), engines=[good, bad])
    assert dp.route() is bad.frontend


def test_all_failed_falls_back_to_first() -> None:
    dp = DPServer(config=object(), engines=[mk(0, "failed"), mk(1, "loading")])
    assert dp.route() is dp.engines[0].frontend


def test_maintenance_state_aggregation() -> None:
    assert DPServer(config=object(), engines=[mk(0, "serving"), mk(1, "loading")]).maintenance_state == "serving"
    assert DPServer(config=object(), engines=[mk(0, "failed"), mk(1, "loading")]).maintenance_state == "loading"
    assert DPServer(config=object(), engines=[mk(0, "failed"), mk(1, "failed")]).maintenance_state == "failed"


def test_fatal_only_when_all_dead() -> None:
    assert DPServer(config=object(), engines=[mk(0, "failed"), mk(1, "serving")]).fatal_error is None
    assert DPServer(config=object(), engines=[mk(0, "failed"), mk(1, "failed")]).fatal_error


def test_backend_processes_and_gpus_aggregate() -> None:
    a, b = mk(0), mk(1)
    a.frontend.backend_processes = ["p0"]
    b.frontend.backend_processes = ["p1", "p2"]
    a.frontend.gpus = [{"rank": 0}]
    b.frontend.gpus = [{"rank": 0}]
    dp = DPServer(config=object(), engines=[a, b])
    assert dp.backend_processes == ["p0", "p1", "p2"]
    assert len(dp.gpus) == 2


def test_sticky_key_forms() -> None:
    assert sticky_key(None) is None
    assert sticky_key({"messages": []}) is None
    assert sticky_key("abc") == sticky_key("abc")
    assert sticky_key({"conversation_id": "c1"}) == "cid:c1"
    k1 = sticky_key({"messages": [{"role": "user", "content": "x"}]})
    k2 = sticky_key({"messages": [{"role": "user", "content": "x"}]})
    assert k1 == k2 and k1.startswith("sha1:")
    assert sticky_key({"prompt": "hi"}) != sticky_key({"prompt": "ho"})


def test_getattr_falls_through_to_engine0() -> None:
    dp = DPServer(config=object(), engines=[mk(0)])
    dp.engines[0].frontend.custom_thing = 42
    assert dp.custom_thing == 42


def test_engine_of() -> None:
    a, b = mk(0), mk(1)
    dp = DPServer(config=object(), engines=[a, b])
    assert dp.engine_of(b.frontend) is b
    assert dp.engine_of(object()) is None


# -- argument-level: DP is off by default and per-engine namespaces are isolated ---------------

ANON_PATH = "/models/anon"


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(argv):
    from unittest.mock import patch

    from freetoken.server.args import parse_args

    config = _Config({"architectures": ["Qwen4ExpForConditionalGeneration"], "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        return parse_args(["--model", ANON_PATH, *argv])


def test_per_engine_address_isolation() -> None:
    base, _ = _parse([])
    e0 = dataclasses.replace(base, dp_index=0, _unique_suffix=".pid=1.dp=0")
    e1 = dataclasses.replace(base, dp_index=1, _unique_suffix=".pid=1.dp=1")
    assert e0.zmq_frontend_addr != e1.zmq_frontend_addr
    assert e0.zmq_backend_addr != e1.zmq_backend_addr
    assert e0.zmq_detokenizer_addr != e1.zmq_detokenizer_addr
    assert e0.distributed_addr != e1.distributed_addr


def test_default_is_single_engine() -> None:
    args, _ = _parse([])
    assert args.data_parallel == 1
    assert args.dp_index == 0


def test_dp_and_tp_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parse(["--data-parallel", "2", "--tensor-parallel-size", "2"])


def test_dp_needs_one_gpu_per_engine() -> None:
    with pytest.raises(SystemExit):
        _parse(["--data-parallel", "2", "--gpu", "0,1,2"])


def test_dp_zero_rejected() -> None:
    with pytest.raises(SystemExit):
        _parse(["--data-parallel", "0"])


def test_dp_accepts_one_gpu_per_engine_and_auto() -> None:
    args, _ = _parse(["--data-parallel", "2", "--gpu", "0,1"])
    assert args.data_parallel == 2 and args.gpu == ("0", "1")
    args2, _ = _parse(["--data-parallel", "2"])
    assert args2.data_parallel == 2 and args2.gpu == ()


def test_routing_stats_counters() -> None:
    dp = DPServer(config=object(), engines=[mk(0), mk(1)])
    dp.route({"messages": [{"role": "user", "content": "a"}]})
    dp.route({"messages": [{"role": "user", "content": "b"}]})
    dp.route({"messages": [{"role": "user", "content": "a"}]})  # sticky hit
    rs = dp.routing_stats()
    assert sum(rs["routed"].values()) == 3
    assert rs["sticky_hits"] == 1
    assert rs["sticky_misses"] == 2


def test_engine_gpu_from_config_fallback() -> None:
    import types

    from freetoken.server.dp import EngineRuntime

    rt = EngineRuntime(index=0, config=types.SimpleNamespace(gpu_assigned=("GPU-abc",), gpu=()),
                       frontend=FakeFM())
    assert rt.gpu()["uuid"] == "GPU-abc"


def test_engine_gpu_from_meta() -> None:
    rt = mk(0)
    rt.frontend.gpus = [{"index": 1, "name": "RTX 3090", "uuid": "GPU-xyz", "total_bytes": 24}]
    g = rt.gpu()
    assert g["uuid"] == "GPU-xyz" and g["name"] == "RTX 3090" and g["rank"] == 0


def test_as_health_reports_gpu_and_restarts() -> None:
    rt = mk(0)
    rt.restarts = 3
    rt.last_failure = "boom"
    h = rt.as_health()
    assert h["restarts"] == 3 and h["last_failure"] == "boom" and "gpu" in h


def test_restart_delay_backoff() -> None:
    from freetoken.server.dp import restart_delay

    assert restart_delay(1) == 1.0
    assert restart_delay(2) == 2.0
    assert restart_delay(3) == 4.0
    assert restart_delay(99, cap_s=30.0) == 30.0
