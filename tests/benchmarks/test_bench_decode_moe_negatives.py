"""Negative fixtures for the decode/spec bench harness (benchmarks/bench_decode_moe.py).

The bench must fail loudly and specifically on the four bad outcomes the plan calls out:
a request timeout, a server crash during startup, an empty generation, and a stream that
ends without a usage chunk. These tests pin those paths without a GPU or a server by
exercising the pure helpers and the failure branches directly.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import urllib.error

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "bench_decode_moe", ROOT / "benchmarks" / "bench_decode_moe.py"
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


class _FakeResp:
    """A minimal SSE response: context manager + byte-line iterable."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


# --- empty output -----------------------------------------------------------------

def test_quality_prefix_empty_plain():
    assert bench.quality_prefix("", "anything") == (0, 0.0)


def test_quality_prefix_empty_spec():
    assert bench.quality_prefix("anything", "") == (0, 0.0)


def test_quality_prefix_partial():
    n, ratio = bench.quality_prefix("abcdef", "abcxyz")
    assert n == 3 and ratio == pytest.approx(0.5)


# --- missing usage (usage) ---------------------------------------------------------

def test_stream_generate_requires_usage_chunk(monkeypatch):
    """A stream with tokens but no usage chunk must die, not silently score 0 tokens."""
    lines = [
        b"data: " + b'{"choices":[{"delta":{"content":"Hi"}}]}' + b"\n",
        b"data: [DONE]\n",
    ]
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **k: _FakeResp(lines))
    args = type("A", (), {"decode": 4})()
    with pytest.raises(SystemExit, match="usage"):
        bench.stream_generate("http://x", "m", "p", {"temperature": 0.0}, args)


def test_stream_generate_http_error(monkeypatch):
    """A non-200 response dies with the HTTP code and body, not a traceback dump."""
    def boom(*a, **k):
        raise urllib.error.HTTPError("http://x", 503, "unavailable", {}, None)

    monkeypatch.setattr(bench.urllib.request, "urlopen", boom)
    args = type("A", (), {"decode": 4})()
    with pytest.raises(SystemExit, match="HTTP 503"):
        bench.stream_generate("http://x", "m", "p", {"temperature": 0.0}, args)


# --- server crash during startup --------------------------------------------------

def test_wait_ready_server_exited(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("boom\n")
    dead = subprocess.Popen(["true"])
    dead.wait()
    with pytest.raises(SystemExit, match="exited with code"):
        bench.wait_ready("http://127.0.0.1:1", dead, str(log), timeout=5.0)


def test_wait_ready_status_error(monkeypatch, tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("boom\n")
    monkeypatch.setattr(bench, "get_json", lambda *a, **k: {"status": "error"})
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(SystemExit, match="startup error"):
            bench.wait_ready("http://127.0.0.1:1", alive, str(log), timeout=5.0)
    finally:
        alive.kill()


# --- timeout ----------------------------------------------------------------------

def test_wait_ready_timeout(tmp_path):
    log = tmp_path / "serve.log"
    log.write_text("still loading\n")
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(SystemExit, match="not ready after"):
            bench.wait_ready("http://127.0.0.1:1", alive, str(log), timeout=0.1)
    finally:
        alive.kill()


# --- counters / problem loading ---------------------------------------------------

def test_parse_spec_acceptance_absent(tmp_path):
    log = tmp_path / "plain.log"
    log.write_text("no spec counters here\n")
    assert bench.parse_spec_acceptance(str(log)) is None


def test_parse_spec_acceptance_zero_proposed(tmp_path):
    log = tmp_path / "spec.log"
    log.write_text("spec accept: 0/0\n")
    out = bench.parse_spec_acceptance(str(log))
    assert out == {"accepted": 0, "proposed": 0, "rate": 0.0}


def test_load_problem_bad_index(tmp_path):
    data = tmp_path / "aime.jsonl"
    data.write_text('{"problem": "q", "answer": "1"}\n')
    with pytest.raises(SystemExit, match="out of range"):
        bench.load_problem(str(data), index=5)


def test_resolve_sampling_greedy():
    params, source = bench.resolve_sampling("/nonexistent-model", greedy=True)
    assert params["temperature"] == 0.0 and "greedy" in source
