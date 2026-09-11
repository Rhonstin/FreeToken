#!/usr/bin/env python3
"""45: concurrency check for --max-running-requests N.

Fires N simultaneous greedy streams at the same server and records per-request
TTFT/decode tok/s/output sha plus the aggregate wall throughput. A correct N=2 run
serves both requests without queueing one behind the other for its whole duration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import threading
import time
import urllib.request


def load_prompts(path: str, n: int) -> list:
    """One distinct prompt per request when the file has >= n lines; otherwise reuse."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [rows[i % len(rows)]["prompt"] for i in range(n)]


def one(origin: str, model: str, prompt: str, max_tokens: int, idx: int, out: list) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0, "top_p": 1.0, "top_k": -1}
    req = urllib.request.Request(f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    usage = None
    text = []
    try:
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
                    tok = d.get("reasoning_content") or d.get("content")
                    if tok:
                        if t_first is None:
                            t_first = time.perf_counter()
                        text.append(tok)
    except Exception as e:  # noqa: BLE001
        out.append({"idx": idx, "error": f"{type(e).__name__}: {e}"})
        return
    t_end = time.perf_counter()
    s = "".join(text)
    out.append({
        "idx": idx,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "ttft_ms": round((t_first - t0) * 1000, 1) if t_first else None,
        "total_s": round(t_end - t0, 3),
        "decode_tok_s": round((len(text) - 1) / (t_end - t_first), 2) if t_first and len(text) > 1 else None,
        "output_sha1": hashlib.sha1(s.encode()).hexdigest()[:12],
        "finish": s[:40],
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="http://127.0.0.1:1919")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--prompt-file", default="/opt/FreeToken/evidence/x299/03b/fixtures/prompt_16k.jsonl")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = load_prompts(a.prompt_file, a.n)
    out: list = []
    th = [threading.Thread(target=one, args=(a.origin, a.model, prompts[i], a.decode, i, out)) for i in range(a.n)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    out.sort(key=lambda r: r["idx"])
    done = [r for r in out if "error" not in r]
    agg = sum((r.get("completion_tokens") or 1) - 1 for r in done) / wall if done else None
    rec = {"schema_version": 2, "type": "parallel", "tag": a.tag, "n": a.n,
           "wall_s": round(wall, 3), "all_ok": len(done) == a.n,
           "aggregate_tok_s": round(agg, 2) if agg else None, "requests": out}
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
