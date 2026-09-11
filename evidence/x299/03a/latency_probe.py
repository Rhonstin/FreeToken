#!/usr/bin/env python3
"""03a latency probe: client-side SSE decode timing against a running FreeToken server.

Reproduction evidence for the decode-latency contract. It measures what a client
sees, per the BENCHMARK_CONTRACT: committed output tokens/s, TPOT, per-event
inter-token latency percentiles and TTFT, plus the server's own reported counters.

Raw rows are emitted as JSONL so they stay immutable. An SSE event is NOT one
token: the numerator uses the committed completion_tokens from the usage chunk,
the denominator spans first-to-last token event.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def stream(origin: str, model_id: str, prompt: str, max_tokens: int, greedy: bool):
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if greedy:
        body.update({"temperature": 0.0, "top_p": 1.0, "top_k": -1})
    req = urllib.request.Request(f"{origin}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps: list[float] = []
    usage = None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content") or delta.get("content"):
                    stamps.append(time.perf_counter())
    return {"t0": t0, "stamps": stamps, "usage": usage}


def pct(sorted_ms: list[float], p: float):
    if not sorted_ms:
        return None
    return sorted_ms[min(len(sorted_ms) - 1, int(len(sorted_ms) * p))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="http://127.0.0.1:1919")
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--prompt", default="Write a detailed multi-paragraph explanation of "
                                        "how a modern CPU executes a program, from fetch to retire.")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--jsonl", required=True)
    a = ap.parse_args()

    model_id = get_json(f"{a.origin}/v1/models")["data"][0]["id"]
    stats_before = get_json(f"{a.origin}/v1/stats")
    warm = stream(a.origin, model_id, a.prompt, min(16, a.decode), a.greedy)  # warm
    r = stream(a.origin, model_id, a.prompt, a.decode, a.greedy)
    stats_after = get_json(f"{a.origin}/v1/stats")

    if r["usage"] is None:
        raise SystemExit("no usage chunk")
    stamps, usage = r["stamps"], r["usage"]
    completion = usage["completion_tokens"]
    steps = completion - 1
    span = stamps[-1] - stamps[0] if len(stamps) >= 2 else 0.0
    gaps = sorted((b - x) * 1e3 for x, b in zip(stamps, stamps[1:]))
    row = {
        "schema_version": 1,
        "origin": a.origin,
        "model_id": model_id,
        "greedy": a.greedy,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": completion,
        "events": len(stamps),
        "decode_steps": steps,
        "decode_tok_s": (steps / span) if span > 0 else None,
        "ms_per_token": (span / steps * 1e3) if steps > 0 else None,
        "ttft_ms": (stamps[0] - r["t0"]) * 1e3 if stamps else None,
        "ttft_warm_first_ms": (warm["stamps"][0] - warm["t0"]) * 1e3 if warm["stamps"] else None,
        "event_ms_p50": pct(gaps, 0.50),
        "event_ms_p95": pct(gaps, 0.95),
        "event_ms_p99": pct(gaps, 0.99),
        "event_ms_max": gaps[-1] if gaps else None,
        "numerator": "completion_tokens - 1 (committed output tokens, excludes first token)",
        "denominator": "last_token_event_ts - first_token_event_ts (seconds)",
        "note": "an SSE event != a token; percentiles are over client token events",
        "server_stats_before": {"decode_tps": stats_before.get("throughput", {}).get("decode_tps"),
                                "ttft_mean_ms": stats_before.get("requests", {}).get("ttft_mean_ms"),
                                "p95_ms": stats_before.get("requests", {}).get("p95_ms")},
        "server_stats_after": {"decode_tps": stats_after.get("throughput", {}).get("decode_tps"),
                               "ttft_mean_ms": stats_after.get("requests", {}).get("ttft_mean_ms"),
                               "p95_ms": stats_after.get("requests", {}).get("p95_ms"),
                               "completion_tokens_total": stats_after.get("requests", {}).get("completion_tokens_total")},
    }
    with open(a.jsonl, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
