#!/usr/bin/env python3
"""48: GSM8K greedy eval of the decode path (offload vs hybrid) against a server.

For each item, one greedy chat completion (reasoning_effort configurable), grade the
final numeric answer, and write per-item rows + a summary with accuracy. Paired across
servers so the offload and hybrid accuracies can be compared item by item.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

PROMPT = ("Solve the math problem. Think step by step, then end your final answer with a "
          "line of the form '#### <number>'.\n\nProblem: {q}")


def last_number(text: str) -> str | None:
    m = re.findall(r"####\s*(-?\d[\d,]*)", text)
    if m:
        return m[-1].replace(",", "").strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    x = nums[-1].replace(",", "").rstrip(".")
    if x.endswith(".0"):
        x = x[:-2]
    return x


def one(origin: str, model: str, q: str, mt: int, effort: str) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
            "max_tokens": mt, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "stream": False}
    if effort:
        body["reasoning_effort"] = effort
    req = urllib.request.Request(f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        resp = json.load(r)
    dt = time.perf_counter() - t0
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = resp.get("usage") or {}
    return {"content": content, "reasoning": reasoning, "usage": usage, "wall_s": round(dt, 2),
            "pred": last_number(content) or last_number(reasoning)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="http://127.0.0.1:1919")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--data", default="/opt/FreeToken/evidence/x299/48/raw/gsm8k.jsonl")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    results = []
    with open(a.out, "a") as outf:
        for r in rows:
            try:
                g = one(a.origin, a.model, r["question"], a.max_tokens, a.effort)
                ok = g["pred"] is not None and g["pred"] == r["gold"]
                rec = {"id": r["id"], "gold": r["gold"], "pred": g["pred"], "correct": ok,
                       "wall_s": g["wall_s"], "completion_tokens": g["usage"].get("completion_tokens"),
                       "prompt_tokens": g["usage"].get("prompt_tokens"),
                       "content_tail": g["content"][-160:], "err": None}
            except Exception as e:  # noqa: BLE001
                rec = {"id": r["id"], "gold": r["gold"], "pred": None, "correct": False,
                       "err": f"{type(e).__name__}: {e}"}
            results.append(rec)
            outf.write(json.dumps({"tag": a.tag, **rec}) + "\n")
            outf.flush()
            print(f"{a.tag} id={rec['id']} gold={rec['gold']} pred={rec['pred']} ok={rec['correct']} "
                  f"ctok={rec.get('completion_tokens')} wall={rec.get('wall_s')}", flush=True)
    n = len(results)
    correct = sum(1 for x in results if x["correct"])
    errs = sum(1 for x in results if x["err"])
    print(f"SUMMARY {a.tag}: {correct}/{n} = {correct / n:.3f} errors={errs}")


if __name__ == "__main__":
    main()
