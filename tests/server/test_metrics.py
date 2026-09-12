"""Tests for the Prometheus metrics surface: the pure formatter, ring percentiles and
the StatsTracker peak/zero pool semantics added for /metrics."""

from __future__ import annotations

from types import SimpleNamespace

from freetoken.server.metrics import gpu_snapshot, to_prometheus
from freetoken.server.request_ring import RequestRecord, RequestRing
from freetoken.server.stats import StatsTracker


def _doc(**over) -> dict:
    doc = {
        "instance_id": "inst",
        "model": {"id": "m", "ctx": 4, "attn": "hybrid_linear", "moe": True},
        "uptime_s": 12,
        "kv": {"used_pages": 448, "total_pages": 220032, "page_size": 1, "quant": "nvfp4"},
        "mamba": {"used_slots": 3, "total_slots": 8},
        "swa": None,
        "vram_bytes": 123,
        "gpus": [{"index": 0, "name": "RTX", "uuid": "GPU-x", "total_bytes": 1}],
        "throughput": {"decode_tps": 30.0, "prefill_tps": 0.0},
        "requests": {"active": 1, "completed": 2, "queued": 3, "p95_ms": 90,
                     "ttft_mean_ms": 300, "prompt_tokens_total": 1000,
                     "cached_tokens_total": 50, "completion_tokens_total": 20},
        "prefill": {"processed_tokens": 500, "total_tokens": 1000, "usage_ratio": 0.5,
                    "eta_seconds": 2.0},
        "spec": {"accepted": 7, "proposed": 10, "rate": 0.7},
        "moe": {"miss_ratio": 0.3, "cache_size": 3447, "hybrid_fetch_fraction": 0.32},
        "gpu_live": [{"index": 0, "uuid": "GPU-x", "name": "RTX", "util_ratio": 0.5,
                      "mem_used_bytes": 10, "mem_total_bytes": 20, "temperature_c": 55,
                      "power_w": 300.0}],
    }
    doc.update(over)
    return doc


def test_to_prometheus_renders_expected_metrics():
    text = to_prometheus(_doc())
    assert "freetoken_info{" in text and 'model="m"' in text
    assert "freetoken_tokens_per_second{phase=\"decode\"} 30.0" in text
    assert "freetoken_kv_pages{kind=\"used\"} 448" in text
    assert "freetoken_kv_usage_ratio 0.00203606" in text or "freetoken_kv_usage_ratio 0.002" in text
    assert "freetoken_mamba_slots{kind=\"used\"} 3" in text
    assert "freetoken_requests_queued 3" in text
    assert "freetoken_prompt_processed_tokens 500" in text
    assert "freetoken_prompt_eta_seconds 2.0" in text
    assert "freetoken_spec_accept_ratio 0.7" in text
    assert "freetoken_moe_cache_miss_ratio 0.3" in text
    assert 'freetoken_gpu_utilization_ratio{index="0"' in text
    assert "freetoken_gpu_power_watts" in text
    # every sample line has a matching TYPE header for its metric
    assert "# TYPE freetoken_kv_pages gauge" in text


def test_to_prometheus_skips_absent_metrics():
    text = to_prometheus(_doc(mamba=None, spec=None, moe=None, prefill=None, gpu_live=[]))
    assert "freetoken_mamba" not in text
    assert "freetoken_spec" not in text
    assert "freetoken_moe" not in text
    assert "freetoken_gpu_" not in text


def test_gpu_snapshot_is_a_list():
    assert isinstance(gpu_snapshot(), list)


def test_request_ring_percentiles():
    ring = RequestRing()
    for i, dur in enumerate([100, 200, 300, 400, 1000]):
        ring.add(RequestRecord(ts="t", method="POST", path="/v1/chat/completions", status=200,
                               model="m", duration_ms=dur, ttft_ms=50, prompt_tokens=10,
                               completion_tokens=100, stream=True, error=None))
    lat = ring.latency()
    assert lat["duration_p50_ms"] == 300
    assert lat["duration_p95_ms"] == 1000
    assert lat["ttft_p50_ms"] == 50
    # decode ms/token = (dur - ttft) / (completion-1) = (100-50)/99 ... p95 is the 1000 case
    assert lat["decode_ms_p95"] == round((1000 - 50) / 99)
    assert lat["n_tokens_max"] == 110


def _reply(**kw):
    base = dict(uid=1, finished=False, completion_tokens_delta=1, prompt_tokens_delta=0,
                cached_tokens=0, kv_used_pages=0, kv_total_pages=0, mamba_used_slots=0,
                mamba_total_slots=0, swa_used_tokens=0, swa_total_tokens=0, gpu_mem_bytes=0,
                queue_reqs=0, prompt_processed=0, prompt_total=0, prefill_active=False,
                spec_accepted=0, spec_proposed=0, moe=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_stats_tracker_peaks_pools_and_resets_at_idle():
    tr = StatsTracker()
    tr.on_new_user(1)
    tr.observe(_reply(kv_used_pages=50, kv_total_pages=100,
                      mamba_used_slots=3, mamba_total_slots=8))
    assert (tr.kv_used_pages, tr.mamba_used_slots) == (50, 3)
    # a 0 snapshot must not clobber the peak while a request is in flight
    tr.observe(_reply(kv_used_pages=0, kv_total_pages=100,
                      mamba_used_slots=0, mamba_total_slots=8))
    assert (tr.kv_used_pages, tr.mamba_used_slots) == (50, 3)
    # finish -> idle -> pools drop to 0
    tr.observe(_reply(finished=True, kv_used_pages=0, kv_total_pages=100,
                      mamba_used_slots=0, mamba_total_slots=8))
    assert tr.active == 0 and tr.completed == 1
    assert (tr.kv_used_pages, tr.mamba_used_slots) == (0, 0)


def test_stats_tracker_prefill_and_counters():
    tr = StatsTracker()
    tr.on_new_user(1)
    tr.observe(_reply(completion_tokens_delta=0, prefill_active=True,
                      prompt_processed=500, prompt_total=1000, queue_reqs=2,
                      cached_tokens=100))
    assert tr.prefill_active and (tr.prompt_processed, tr.prompt_total) == (500, 1000)
    assert tr.queue_reqs == 2 and tr.cached_tokens_total == 100
    tr.observe(_reply(prefill_active=False, spec_accepted=7, spec_proposed=10))
    assert not tr.prefill_active
    assert (tr.spec_accepted, tr.spec_proposed) == (7, 10)
