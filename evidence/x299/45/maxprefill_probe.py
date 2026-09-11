#!/usr/bin/env python3
"""45: full declared-max (~219k token) prefill + sustained decode probe.

Builds a long prompt from repo docs, streams one chat completion, and records the
reported prompt_tokens, TTFT, decode tok/s and post-run VRAM. A second request on the
same text (radix-cached prefill) measures sustained decode at near-max context.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.request


def build_prompt(chars: int) -> str:
    parts = []
    for f in ("/opt/FreeToken/README.md", "/opt/FreeToken/CONTRIBUTING.md", "/opt/FreeToken/docs/install.md"):
        p = pathlib.Path(f)
        if p.is_file():
            parts.append(" ".join(p.read_text(errors="ignore").split()))
    parts.append(
        "The origin of the modern weather forecast lies in the nineteenth century when a severe "
        "storm struck the British Isles and observers began sending simultaneous reports by "
        "telegraph from many stations, and the resulting maps revealed patterns that repeated."
    )
    base = " ".join(parts)
    out = []
    n = 0
    while n < chars:
        out.append(base)
        n += len(base)
    return " ".join(out)[:chars]


def run_once(origin: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0, "top_p": 1.0, "top_k": -1}
    req = urllib.request.Request(f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    usage = None
    ntok = 0
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
                if d.get("reasoning_content") or d.get("content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    ntok += 1
    t_end = time.perf_counter()
    out = {"prompt_tokens": (usage or {}).get("prompt_tokens"),
           "completion_tokens": (usage or {}).get("completion_tokens"),
           "ttft_s": round(t_first - t0, 3) if t_first else None,
           "total_s": round(t_end - t0, 3),
           "decode_tok_s": round((ntok - 1) / (t_end - t_first), 2) if t_first and ntok > 1 else None,
           "chunks": ntok}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="http://127.0.0.1:2061")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--chars", type=int, default=830000)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompt = build_prompt(a.chars)
    res = {"prompt_chars": len(prompt), "runs": []}
    for tag, mt in (("prefill-peak", a.max_tokens), ("sustained-decode", a.max_tokens + 64)):
        r = run_once(a.origin, a.model, prompt, mt)
        r["tag"] = tag
        res["runs"].append(r)
        print(json.dumps(r), flush=True)
        h = json.load(urllib.request.urlopen(f"{a.origin}/health", timeout=10))
        r["health_after"] = h.get("status")
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
