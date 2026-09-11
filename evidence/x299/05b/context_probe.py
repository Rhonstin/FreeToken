#!/usr/bin/env python3
"""05b context probe: greedy decode at 4k/16k/32k against a running server.

Reads prompt_<label>.jsonl (03b fixtures), does one warmup then one measured run
per context, and appends schema-v2 rows. An SSE event is not a token.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def stream(origin, model_id, prompt, max_tokens, greedy):
    body = {"model": model_id, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}}
    if greedy:
        body.update({"temperature": 0.0, "top_p": 1.0, "top_k": -1})
    req = urllib.request.Request(f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps, usage, text = [], None, []
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            p = line[len(b"data:"):].strip()
            if p == b"[DONE]":
                break
            c = json.loads(p)
            if c.get("usage"):
                usage = c["usage"]
            for ch in c.get("choices", []):
                d = ch.get("delta") or {}
                t = d.get("reasoning_content") or d.get("content")
                if t:
                    stamps.append(time.perf_counter())
                    text.append(t)
    return {"t0": t0, "stamps": stamps, "usage": usage or {}, "text": "".join(text)}


def pct(s, p):
    return s[min(len(s) - 1, int(len(s) * p))] if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="http://127.0.0.1:1931")
    ap.add_argument("--prompt-dir", required=True)
    ap.add_argument("--contexts", default="4k,16k,32k")
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--jsonl", required=True)
    a = ap.parse_args()
    model = get_json(f"{a.origin}/v1/models")["data"][0]["id"]
    for label in a.contexts.split(","):
        pfile = pathlib.Path(a.prompt_dir) / f"prompt_{label}.jsonl"
        if not pfile.is_file():
            continue
        prompt = json.loads(pfile.read_text().splitlines()[0])["prompt"]
        stream(a.origin, model, prompt, min(16, a.decode), True)  # warmup
        r = stream(a.origin, model, prompt, a.decode, True)
        u = r["usage"]
        steps = u.get("completion_tokens", len(r["stamps"])) - 1
        span = r["stamps"][-1] - r["stamps"][0] if len(r["stamps"]) >= 2 else 0.0
        gaps = sorted((b - x) * 1e3 for x, b in zip(r["stamps"], r["stamps"][1:]))
        row = {"schema_version": 2, "type": "warm", "mode": a.mode, "context": label,
               "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
               "events": len(r["stamps"]), "decode_steps": steps,
               "decode_tok_s": steps / span if span > 0 else 0.0,
               "ms_per_token": span / steps * 1e3 if steps > 0 else 0.0,
               "ttft_ms": (r["stamps"][0] - r["t0"]) * 1e3 if r["stamps"] else None,
               "event_ms_p50": pct(gaps, 0.50), "event_ms_p95": pct(gaps, 0.95),
               "numerator": "completion_tokens - 1",
               "denominator": "last_token_event_ts - first_token_event_ts (s)",
               "output_sha1": __import__("hashlib").sha1(r["text"].encode()).hexdigest()[:12]}
        with open(a.jsonl, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row))


if __name__ == "__main__":
    main()
