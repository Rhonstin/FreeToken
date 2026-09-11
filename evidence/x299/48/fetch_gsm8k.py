#!/usr/bin/env python3
"""48: fetch the first N GSM8K test rows (HF datasets-server) into a fixed jsonl.

Each row: {"id", "question", "gold"} where gold is the number after ``####``.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request

URL = ("https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k&config=main"
       "&split=test&offset={off}&length={n}")


def gold_of(answer: str) -> str | None:
    m = re.search(r"####\s*(-?[\d,]+)", answer)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d[\d,]*", answer)
    return nums[-1].replace(",", "") if nums else None


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    out = sys.argv[2] if len(sys.argv) > 2 else "evidence/x299/48/raw/gsm8k.jsonl"
    rows = []
    off = 0
    while len(rows) < n:
        take = min(100, n - len(rows))
        with urllib.request.urlopen(URL.format(off=off, n=take), timeout=60) as r:
            data = json.load(r)
        got = data.get("rows", [])
        if not got:
            break
        for i, item in enumerate(got):
            row = item["row"]
            rows.append({"id": off + i, "question": row["question"], "gold": gold_of(row["answer"])})
        off += len(got)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
