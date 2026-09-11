#!/usr/bin/env python3
"""12a PLE row-cache trace: row-ID locality + disk read latency (no C++ changes).

Replicates PleStore.hash_rows exactly (wrap_mul/pos_mod) with the checkpoint's derived
n-gram constants, then measures cross-fill row reuse under the auto fill sizes and the
same through a bounded LRU at 256 MiB / 1 GiB / 2 GiB. Also times random row reads from
the on-disk table. Locality decides whether a bounded RAM row cache can pay off at all.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import OrderedDict

from tokenizers import Tokenizer

M = "/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4"
EOS = 248044
VOCAB = 248320
MASK64 = (1 << 64) - 1


def wrap_mul(a: int, b: int) -> int:
    return ((a & MASK64) * (b & MASK64)) & MASK64


def pos_mod(v: int, m: int) -> int:
    return v % m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="/opt/FreeToken/evidence/x299/03b/fixtures/prompt_16k.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-tokens", type=int, default=16384)
    a = ap.parse_args()

    from freetoken.models.qwen4_exp.ple import derive_ngram_hash_constants
    mult, sizes, offsets = derive_ngram_hash_constants(
        vocab_size=VOCAB, ngram_size=3, num_ngram_heads=16,
        ngram_vocab_size_base=20000000, ple_layer_index=0)
    heads = len(sizes)
    half = heads // 2
    row_bytes = 2560 // heads

    def hash_rows(t0: int, t1: int, t2: int) -> list[int]:
        prev1 = t1
        prev2 = EOS if prev1 == EOS else t0
        bigram = wrap_mul(t2, mult[0]) ^ wrap_mul(prev1, mult[1])
        trigram = bigram ^ wrap_mul(prev2, mult[2])
        return [pos_mod(bigram if h < half else trigram, sizes[h]) + offsets[h] for h in range(heads)]

    def rows_for_range(tokens: list[int], start: int, end: int) -> list[int]:
        out: list[int] = []
        for k in range(start, end):
            t0 = tokens[k - 2] if k >= 2 else EOS
            t1 = tokens[k - 1] if k >= 1 else EOS
            out.extend(hash_rows(t0, t1, tokens[k]))
        return out

    def fills_for(tokens: list[int], fill_tokens: int) -> list[list[int]]:
        out = []
        k = 2
        while k < len(tokens):
            nxt = min(k + fill_tokens, len(tokens))
            out.append(rows_for_range(tokens, k, nxt))
            k = nxt
        return out

    prompt = json.loads(open(a.prompt).read().splitlines()[0])["prompt"]
    tok = Tokenizer.from_file(f"{M}/tokenizer.json")
    real = tok.encode(prompt).ids
    rng = random.Random(0)
    rand = [rng.randrange(VOCAB) for _ in range(a.random_tokens)]

    def stats(fills: list[list[int]], label: str) -> dict:
        total = sum(len(f) for f in fills)
        seen: set[int] = set()
        hits = 0
        for f in fills:
            for rid in f:
                if rid in seen:
                    hits += 1
            seen.update(f)
        return {"label": label, "fills": len(fills), "lookups": total, "unique_total": len(seen),
                "within_fill_dup": total - sum(len(set(f)) for f in fills),
                "cross_fill_hits_infinite": hits,
                "cross_fill_hit_rate_infinite": round(hits / total, 6) if total else None}

    def lru(fills: list[list[int]], cap_rows: int):
        cache: "OrderedDict[int, None]" = OrderedDict()
        hits = total = 0
        for f in fills:
            for rid in f:
                total += 1
                if rid in cache:
                    hits += 1
                    cache.move_to_end(rid)
                else:
                    cache[rid] = None
                    if len(cache) > cap_rows:
                        cache.popitem(last=False)
        return round(hits / total, 6) if total else None

    dec_f = fills_for(real, 1)
    pre_f = fills_for(real, 8192)
    rand_f = fills_for(rand, 1)
    caps = {name: int(mib * 2**20 / row_bytes) for name, mib in (("256MiB", 256), ("1GiB", 1024), ("2GiB", 2048))}

    out = {
        "constants": {"mult": mult, "num_heads": heads, "sizes_head0": sizes[0],
                      "row_bytes": row_bytes, "table_rows": offsets[-1] + sizes[-1],
                      "table_gib": round((offsets[-1] + sizes[-1]) * row_bytes / 2**30, 2)},
        "real_decode": stats(dec_f, "real decode (1 tok/fill)"),
        "real_prefill": stats(pre_f, "real prefill (8192 tok/fill)"),
        "random_decode": stats(rand_f, "random decode (1 tok/fill)"),
        "lru_capacity_rows": caps,
        "lru_hit_rate": {
            "real_decode": {k: lru(dec_f, v) for k, v in caps.items()},
            "real_prefill": {k: lru(pre_f, v) for k, v in caps.items()},
            "random_decode": {k: lru(rand_f, v) for k, v in caps.items()},
        },
    }

    try:
        from freetoken.models.qwen4_exp.ple_disk import resolve_row_source
        src = resolve_row_source(M)
        fds: dict[str, int] = {}

        def read_one(rid: int) -> None:
            ext = rid // src.rows_per_extent
            path = src.paths[src.extent_file[ext]]
            fd = fds.get(path)
            if fd is None:
                fd = fds[path] = os.open(path, os.O_RDONLY)
            off = src.extent_base[ext] + (rid % src.rows_per_extent) * src.row_stride
            os.pread(fd, src.row_bytes, off)

        picks = [rng.randrange(src.total_rows) for _ in range(500)]
        warm = []
        for _ in range(3):
            t0 = time.perf_counter()
            for rid in picks:
                read_one(rid)
            warm.append((time.perf_counter() - t0) / len(picks) * 1e6)
        for fd in fds.values():
            os.close(fd)
        out["disk_latency_us_per_random_row"] = {
            "warm_median": round(sorted(warm)[len(warm) // 2], 2),
            "runs": [round(w, 2) for w in warm], "row_bytes": src.row_bytes}
    except Exception as e:  # noqa: BLE001
        out["disk_latency_us_per_random_row"] = {"error": repr(e)}

    print(json.dumps(out, indent=2))
    open(a.out, "w").write(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
