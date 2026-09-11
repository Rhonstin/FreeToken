#!/usr/bin/env python3
"""41: teacher-forced PPL/top1 corpus build + scoring harness.

Builds a fixed, non-repetitive corpus from repo docs and scores it via POST /v1/score
(no chat template) against whatever server is at --origin. Used to compare offload vs
hybrid MoE quality.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import urllib.request

FILES = [
    "/opt/FreeToken/README.md",
    "/opt/FreeToken/CONTRIBUTING.md",
    "/opt/FreeToken/docs/install.md",
    "/opt/FreeToken/docs/quickstart.md",
    "WEATHER",
]
WEATHER = (
    "The origin of the modern weather forecast lies in the nineteenth century, when a severe "
    "storm struck the British Isles and prompted a public demand for advance warning. Observers "
    "began sending simultaneous reports by telegraph from many stations, and the resulting maps "
    "revealed patterns that repeated from day to day. As the network spread across the continent, "
    "forecasters learned to track moving pressure systems rather than memorize local signs. By the "
    "early twentieth century, mathematical models of the atmosphere had been proposed, though the "
    "arithmetic required to solve them by hand proved impossible. The electronic computer changed "
    "that. A small team ran the first numerical prediction on a machine with almost no memory, and "
    "the result agreed well enough with the actual weather to justify years of further work. Modern "
    "forecasts now blend billions of observations with models that run on the fastest machines "
    "available, and the same idea reaches from the daily outlook on a phone to warnings issued "
    "before a hurricane reaches land."
)


def build_corpus(out: pathlib.Path, chars: int) -> None:
    rows = []
    for f in FILES:
        if f == "WEATHER":
            text = WEATHER
        else:
            p = pathlib.Path(f)
            if not p.is_file():
                continue
            text = p.read_text(errors="ignore")
        text = " ".join(text.split())
        for i in range(0, min(len(text), chars * 3), chars):
            chunk = text[i : i + chars].strip()
            if len(chunk) > 500:
                rows.append({"id": f"{pathlib.Path(f).name}#{i}", "text": chunk})
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"corpus: {len(rows)} passages -> {out}")


def score(origin: str, text: str, chunk: int) -> dict:
    req = urllib.request.Request(
        f"{origin}/v1/score",
        data=json.dumps({"text": text, "chunk": chunk}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default="http://127.0.0.1:1919")
    ap.add_argument("--corpus", default="/opt/FreeToken/evidence/x299/41/corpus.jsonl")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--chars", type=int, default=6000)
    ap.add_argument("--tag")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.build:
        build_corpus(pathlib.Path(a.corpus), a.chars)
        return
    assert a.tag and a.out, "--tag/--out required for scoring"
    rows = [json.loads(l) for l in pathlib.Path(a.corpus).read_text().splitlines() if l.strip()]
    results, toks = [], 0
    for r in rows:
        b = score(a.origin, r["text"], 1024)
        nll = b["nll"]
        assert nll and all(math.isfinite(x) for x in nll), r["id"]
        assert len(nll) == b["n_tokens"] - 1
        assert abs(b["ppl"] - math.exp(sum(nll) / len(nll))) < 1e-9
        results.append({"id": r["id"], "n_tokens": b["n_tokens"], "ppl": b["ppl"],
                        "top1_rate": b["top1_rate"], "mean_nll": sum(nll) / len(nll)})
        toks += b["n_tokens"]
    agg_ppl = math.exp(sum(x["mean_nll"] * x["n_tokens"] for x in results) / toks)
    out = {"tag": a.tag, "passages": len(results), "tokens": toks,
           "agg_ppl": agg_ppl, "mean_top1_rate": sum(x["top1_rate"] for x in results) / len(results),
           "rows": results}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("tag", "passages", "tokens", "agg_ppl", "mean_top1_rate")}, indent=2))


if __name__ == "__main__":
    main()
