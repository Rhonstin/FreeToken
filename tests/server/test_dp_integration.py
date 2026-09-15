"""Integration test: run_api_server assembles a DP serve from N engine handles.

No CUDA, no ZMQ, no worker processes — the ZMQ queues, the backend launcher, the supervisor
and uvicorn are all faked, so this exercises the real assembly code (DPServer construction,
per-engine wiring, routing, aggregate control plane) end to end through HTTP."""

from __future__ import annotations

import asyncio
import types

import pytest
from fastapi.testclient import TestClient

from freetoken.server import api_server as A
from freetoken.server.dp import DPServer


def _args(i: int):
    model_config = types.SimpleNamespace(
        has_linear_attention=False, has_swa_attention=False, is_moe=True,
        num_experts=8, num_moe_layers=2,
    )
    return types.SimpleNamespace(
        zmq_frontend_addr=f"ipc:///tmp/test_front_{i}",
        zmq_tokenizer_addr=f"ipc:///tmp/test_tok_{i}",
        frontend_create_tokenizer_link=False,
        model_path="/nonexistent",
        served_model_name="m",
        max_seq_len=4096,
        page_size=1,
        kv_quant="none",
        moe_cache_policy="lru",
        model_config=model_config,
        gpu_assigned=(f"GPU-{i}",),
        gpu=(),
    )


class FakeHandle:
    def __init__(self, i: int) -> None:
        self.config = _args(i)
        self.processes = []
        self.expected_acks = 0
        self.restart = None
        self.ack_queue = None


class FakePush:
    def __init__(self, *a, **k) -> None:
        self.stopped = False

    async def put(self, _item):
        return None

    def stop(self) -> None:
        self.stopped = True


class FakePull:
    def __init__(self, *a, **k) -> None:
        pass

    async def get(self):
        await asyncio.sleep(3600)

    def stop(self) -> None:
        return None


@pytest.fixture
def dp_app(monkeypatch):
    monkeypatch.setattr(A, "_GLOBAL_STATE", None)
    A._SHUTTING_DOWN.clear()
    monkeypatch.setattr(A, "install_cors", lambda *a, **k: None)
    monkeypatch.setattr(A, "init_request_logging", lambda *a, **k: None)
    monkeypatch.setattr(A, "install_polling_access_log_filter", lambda *a, **k: None)
    monkeypatch.setattr(A, "ZmqAsyncPullQueue", FakePull)
    monkeypatch.setattr(A, "ZmqAsyncPushQueue", FakePush)
    monkeypatch.setattr(
        "freetoken.server.supervisor.run_backend_supervisor", lambda *a, **k: None
    )
    monkeypatch.setattr(A.uvicorn, "run", lambda *a, **k: None)
    yield
    A._GLOBAL_STATE = None
    A._SHUTTING_DOWN.clear()


def _run(monkeypatch, n=2):
    handles = [FakeHandle(i) for i in range(n)]
    monkeypatch.setattr(A, "load_generation_sampling", lambda *a, **k: {})
    config = types.SimpleNamespace(
        sampling_defaults="none", use_dummy_weight=False, server_host="127.0.0.1",
        server_port=1919, cors_origins="", model_path="/nonexistent",
        served_model_name="m", data_parallel=n,
    )
    A.run_api_server(config, lambda: handles, run_shell=False)
    return handles


def test_run_api_server_builds_n_engines(dp_app, monkeypatch):
    _run(monkeypatch, n=2)
    dp = A._GLOBAL_STATE
    assert isinstance(dp, DPServer)
    assert [e.index for e in dp.engines] == [0, 1]
    fms = [e.frontend for e in dp.engines]
    assert fms[0] is not fms[1]
    # routing returns one of the two engines
    routed = A.get_global_state({"messages": [{"role": "user", "content": "x"}]})
    assert any(routed is fm for fm in fms)
    assert A.get_dp_state() is dp


def test_dp_control_endpoints_see_all_engines(dp_app, monkeypatch):
    _run(monkeypatch, n=2)
    with TestClient(A.app) as client:
        health = client.get("/health").json()
        assert health["data_parallel"] == {"size": 2, "serving": 0}
        assert len(health["engines"]) == 2
        dp = client.get("/v1/dp/status").json()
        assert dp["mode"] == "data_parallel"
        assert dp["data_parallel"]["size"] == 2
        assert len(dp["engines"]) == 2
        assert "arena" in dp
        metrics = client.get("/metrics").text
        assert "freetoken_dp_engines 2" in metrics


def test_single_engine_is_one_dp_engine(dp_app, monkeypatch):
    _run(monkeypatch, n=1)
    dp = A._GLOBAL_STATE
    assert len(dp.engines) == 1
    with TestClient(A.app) as client:
        assert client.get("/v1/dp/status").json()["mode"] == "single"
