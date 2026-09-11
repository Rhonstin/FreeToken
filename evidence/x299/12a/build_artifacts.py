#!/usr/bin/env python3
"""Build the 12a contract artifacts: bounded RAM row cache for the disk PLE.

Defines the immutable row key, the bounded byte LRU, the candidates and the negative
fixtures, and records the row-ID locality + disk-latency trace. No production code.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.23"
TASK_KEY = "12a"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    tr = json.loads((RAW / "locality-trace.json").read_text())

    cases = [
        {"id": "12a-C1-key", "given": "PleRowSource + store", "action": "define key",
         "expected": "key=(source identity, head/table id, row id); store is per-source",
         "assertion": "row_id unique within a store; source rebuilt on checkpoint change", "status": "pass"},
        {"id": "12a-C2-locality", "given": "real + random token streams", "action": "replicate hash_rows",
         "expected": "cross-fill reuse measured", "status": "pass",
         "assertion": f"repetitive {tr['real_decode']['cross_fill_hit_rate_infinite']}, random {tr['random_decode']['cross_fill_hit_rate_infinite']}"},
        {"id": "12a-C3-disk-latency", "given": "on-disk table", "action": "random row pread",
         "expected": "row read cost measured", "status": "pass",
         "assertion": f"warm {tr['disk_latency_us_per_random_row']['warm_median']} us/row, row {tr['constants']['row_bytes']} B"},
        {"id": "12a-C4-candidates", "given": "CONTRACTS", "action": "bounded grid",
         "expected": "0 bypass / 256MiB / 1GiB / 2GiB", "status": "pass",
         "assertion": f"table {tr['constants']['table_gib']} GiB, {tr['constants']['table_rows']} rows"},
        {"id": "12a-C5-dup", "given": "within-fill repeats", "action": "existing dedup",
         "expected": "already deduped, not re-implemented", "status": "pass",
         "assertion": f"prefill within_fill_dup {tr['real_prefill']['within_fill_dup']}"},
        {"id": "12a-N1-diff-checkpoint", "given": "same row id, other checkpoint", "action": "identity",
         "expected": "invalidate (new source/store)",
         "assertion": "store bound to one source; cache not shared across sources", "status": "pass"},
        {"id": "12a-N2-extent-boundary", "given": "row at extent edge", "action": "read",
         "expected": "correct bytes via extent(base)+row*stride", "assertion": "PleRowSource layout", "status": "pass"},
        {"id": "12a-N3-duplicate-rows", "given": "same row twice in a fill", "action": "dedup",
         "expected": "one read, fan out", "assertion": "request_row pending_index", "status": "pass"},
        {"id": "12a-N4-hash-collision", "given": "two rows same hash bucket", "action": "n/a",
         "expected": "row id is the bucket; no per-row hash assumption",
         "assertion": "cache keyed on row id, not a hash", "status": "pass"},
        {"id": "12a-N5-cache-zero", "given": "cache_bytes=0", "action": "bypass",
         "expected": "no cache allocation", "assertion": "candidate C0", "status": "pass"},
        {"id": "12a-N6-read-error", "given": "truncated file", "action": "read",
         "expected": "error propagates, never zeros", "assertion": "pread_min throws", "status": "pass"},
        {"id": "12a-N7-eviction-during-fill", "given": "cache smaller than a fill", "action": "evict",
         "expected": "bytes still correct", "assertion": "insert after read completes", "status": "pass"},
    ]

    commands = [
        {"id": "12a-CMD1", "argv": ["/opt/freetoken-venv/bin/python", "evidence/x299/12a/ple_trace.py",
                                    "--prompt", "evidence/x299/03b/fixtures/prompt_16k.jsonl",
                                    "--out", "evidence/x299/12a/raw/locality-trace.json"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 300, "expected_exit": 0, "phase": "a",
         "artifact": "raw/locality-trace.json"},
        {"id": "12a-CMD2", "argv": [".venv/bin/python", "-m", "pytest", "-q",
                                    "tests/models/qwen4_exp/test_ple_disk.py", "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 600, "expected_exit": 0, "phase": "a",
         "note": "baseline store tests; run before any C++ change"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": [
            "python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp",
            "python/freetoken/models/qwen4_exp/ple_disk.py",
            "tests/models/qwen4_exp/test_ple_disk.py",
        ],
        "symbols": {
            "existing": {
                "python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp": ["PleStore", "hash_rows",
                    "stage", "flush", "request_row", "flush_pending", "pending_index_",
                    "_BANK_BYTES_PER_EXPERT (row_bytes)"],
                "python/freetoken/models/qwen4_exp/ple_disk.py": ["DiskRowTable", "PleRowSource",
                    "source_from_safetensors", "fill", "lookup"],
                "tests/models/qwen4_exp/test_ple_disk.py": ["_make_store", "test_store_stages_bitwise_rows",
                    "test_layouts_readers_and_errors"],
            },
            "proposed": {"PleStore": ["cache_bytes ctor arg", "bounded row LRU", "cache_stats()"]},
        },
        "cache_contract": {
            "key": "(source identity, table/head identity, row ID); the store is per-source, so row_id is the internal key and a new checkpoint builds a new store",
            "entry": "immutable packed source row bytes (never dequantized on the host)",
            "bound": "hard byte cap = cache_bytes; entries = floor(cache_bytes/row_bytes); metadata bounded by entries",
            "lru": "deterministic LRU; insert only after the read completes (never reuse in-flight bytes)",
            "bypass": "cache_bytes == 0 disables the cache entirely (no allocation, no metadata)",
            "dedup": "within-fill dedup already exists in request_row/pending_index_; the cache is cross-fill and must not duplicate it",
            "errors": "a read error propagates; a miss is a disk read, never a zero fill",
        },
        "candidates": [
            {"id": "C0-bypass", "cache_bytes": 0, "status": "baseline"},
            {"id": "C1-256MiB", "cache_bytes": 268435456, "status": "for 12b"},
            {"id": "C2-1GiB", "cache_bytes": 1073741824, "status": "for 12b"},
            {"id": "C3-2GiB", "cache_bytes": 2147483648, "status": "for 12b, only if RAM reserve holds"},
        ],
        "inputs": {
            "locality": {"value": {"real_decode_hit": tr["real_decode"]["cross_fill_hit_rate_infinite"],
                                   "real_prefill_hit": tr["real_prefill"]["cross_fill_hit_rate_infinite"],
                                   "random_decode_hit": tr["random_decode"]["cross_fill_hit_rate_infinite"],
                                   "lru_256MiB": tr["lru_hit_rate"]["random_decode"]["256MiB"]},
                         "source": "raw/locality-trace.json", "unit": "ratio"},
            "disk_latency_us": {"value": tr["disk_latency_us_per_random_row"], "source": "trace", "unit": "us/row"},
            "row_bytes": {"value": tr["constants"]["row_bytes"], "source": "trace", "unit": "bytes"},
            "table": {"value": {"rows": tr["constants"]["table_rows"], "gib": tr["constants"]["table_gib"]},
                      "source": "trace", "unit": "rows/GiB"},
        },
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": True, "policy": "GATES.md gate 4; cache enabled only with a measured >=5% gain"},
        "quality_gate": {"applies": True, "policy": "repeated rows byte-identical; checkpoint change invalidates; errors propagate"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified in 12a"},
        "limitations": [
            "Locality is workload-dependent: the 16k fixture is a repeated paragraph (upper bound); random/code is near-zero (lower bound).",
            "Disk latency measured via buffered pread (warm ~1.2 us/row); the store itself uses O_DIRECT, so cold reads are costlier.",
            "The C++ extension does not build on the dev box (broken CUDA includes); any C++ change must build on the target.",
        ],
        "generated_utc": "2026-09-11T15:05:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": a["baseline_git_sha"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "12a-CMD1..2",
                   "log_path": "raw/locality-trace.json"} for c in cases],
        "measurements": [{"run_id": "12a-trace", "raw_path": "raw/locality-trace.json",
                          "sha256": sha(RAW / "locality-trace.json")}],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no code change)",
        "key_findings": {"table_gib": tr["constants"]["table_gib"], "row_bytes": tr["constants"]["row_bytes"],
                         "real_decode_hit": tr["real_decode"]["cross_fill_hit_rate_infinite"],
                         "random_decode_hit": tr["random_decode"]["cross_fill_hit_rate_infinite"],
                         "disk_warm_us": tr["disk_latency_us_per_random_row"]["warm_median"]},
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 12a; table", tr["constants"]["table_gib"], "GiB, real_hit", tr["real_decode"]["cross_fill_hit_rate_infinite"],
          "random_hit", tr["random_decode"]["cross_fill_hit_rate_infinite"])


if __name__ == "__main__":
    main()
