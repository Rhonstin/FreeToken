#!/usr/bin/env python3
"""19b (38): prefill / multi-turn TTFT probe.

Measures TTFT for (1) a cold short prompt, (2) the same prompt again (radix hit), and
(3) a second turn continuing the conversation (prefix reuse + one new segment).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def stream_ttft(origin: str, model: str, messages: list, max_tokens: int) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.0, "top_p": 1.0, "top_k": -1}
    req = urllib.request.Request(f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    usage = None
    n = 0
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
                if d.get("content") or d.get("reasoning_content"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    n += 1
    t_end = time.perf_counter()
    return {"prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "ttft_ms": round((t_first - t0) * 1000, 1) if t_first else None,
            "total_s": round(t_end - t0, 3),
            "decode_tok_s": round((n - 1) / (t_end - t_first), 2) if t_first and n > 1 else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="http://127.0.0.1:2151")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = ("Explain, step by step, how a modern CPU pipeline handles a branch misprediction, "
            "and what the recovery cost is. Keep it short.")
    out = {"runs": []}
    msgs = [{"role": "user", "content": base}]
    for tag in ("cold", "radix-hit"):
        r = stream_ttft(a.origin, a.model, msgs, 32)
        r["tag"] = tag
        out["runs"].append(r)
        print(json.dumps(r), flush=True)
    # second turn: continue the conversation
    msgs2 = msgs + [{"role": "assistant", "content": "A branch misprediction flushes the pipeline stages."},
                    {"role": "user", "content": "And the cost in cycles?"}]
    r = stream_ttft(a.origin, a.model, msgs2, 32)
    r["tag"] = "turn2"
    out["runs"].append(r)
    print(json.dumps(r), flush=True)
    with open(a.out, "w") as f:
        f.write(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
