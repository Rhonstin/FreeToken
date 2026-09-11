#!/usr/bin/env python3
"""45: write N distinct prompts of ~target token count (chars/3.81).

Each prompt starts with a unique marker and a rotated body so radix cannot share the
prefix; this is the fair input for a shared-KV-pool concurrency test.
"""
import json, pathlib, sys
tokens = int(sys.argv[1]); n = int(sys.argv[2]); out = sys.argv[3]
chars = int(tokens * 3.81)
parts = []
for f in ("/opt/FreeToken/README.md", "/opt/FreeToken/CONTRIBUTING.md", "/opt/FreeToken/docs/install.md"):
    p = pathlib.Path(f)
    if p.is_file():
        parts.append(" ".join(p.read_text(errors="ignore").split()))
base = " ".join(parts)
rows = []
for i in range(n):
    rot = base[i * 1013:] + " " + base[: i * 1013]
    seed = f"UNIQUE-MARKER-{i}-{i*7919} "
    body = seed + rot
    b = body
    while len(b) < chars:
        b += " " + body
    rows.append({"prompt": b[:chars]})
pathlib.Path(out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"wrote {out}: {n} x ~{tokens} tok ({chars} chars)")
