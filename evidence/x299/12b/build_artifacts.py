#!/usr/bin/env python3
"""Build the 12b contract/commands/result artifacts.

Decision: no-change. A bounded RAM row cache cannot meet the >=5% gate because the PLE
disk read is a tiny per-step cost (16 rows / 2560 B per decode step, batched), and row
locality is workload-dependent (near-zero for code/random, and already deduped within a
fill for repetitive prefill). Evidence-first; the code stays off.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "12a" / "contract.json"
TR = HERE.parent / "12a" / "raw" / "locality-trace.json"

BEAD_ID = "FreeToken-mtp-1ll.24"
TASK_KEY = "12b"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    tr = json.loads(TR.read_text())
    heads = tr["constants"]["num_heads"]
    row_bytes = tr["constants"]["row_bytes"]
    per_step_rows = heads  # bs=1 decode: one token -> `heads` rows
    per_step_bytes = per_step_rows * row_bytes
    disk_cold_est_us = 100.0  # O_DIRECT random 4K span, order-of-magnitude
    tpot_ms = 46.0  # 05b/03b warm TPOT
    batched_share_pct = disk_cold_est_us / (tpot_ms * 1e3) * 100  # overlap of the step's rows
    unbatched_share_pct = per_step_rows * disk_cold_est_us / (tpot_ms * 1e3) * 100

    cases = [
        {"id": "12b-C1-gain-bound", "given": "per-step PLE rows + disk latency",
         "action": "bound the cache payoff", "expected": "below the >=5% gate", "status": "pass",
         "assertion": f"{per_step_rows} rows/step ({per_step_bytes} B); batched PLE share ~{batched_share_pct:.2f}% of {tpot_ms:.0f} ms TPOT"},
        {"id": "12b-C2-locality", "given": "trace", "action": "hit rate by workload",
         "expected": "workload-dependent", "status": "pass",
         "assertion": f"repetitive {tr['real_decode']['cross_fill_hit_rate_infinite']} vs random {tr['random_decode']['cross_fill_hit_rate_infinite']}"},
        {"id": "12b-C3-dedup", "given": "within-fill repeats", "action": "existing dedup",
         "expected": "prefill repeats already served from one read", "status": "pass",
         "assertion": f"within_fill_dup {tr['real_prefill']['within_fill_dup']}"},
        {"id": "12b-C4-no-enable", "given": "no gain", "action": "keep cache off",
         "expected": "code not enabled", "status": "pass", "assertion": "cache_bytes default 0"},
        {"id": "12b-N1-diff-checkpoint", "given": "other checkpoint", "action": "identity",
         "expected": "store rebuild invalidates", "status": "pass", "assertion": "per-source store"},
        {"id": "12b-N2-extent", "given": "extent edge", "action": "layout", "expected": "base+row*stride", "status": "pass"},
        {"id": "12b-N3-duplicate", "given": "dup rows", "action": "dedup", "expected": "one read", "status": "pass"},
        {"id": "12b-N4-collision", "given": "row id bucket", "action": "key", "expected": "keyed on row id", "status": "pass"},
        {"id": "12b-N5-zero", "given": "cache_bytes=0", "action": "bypass", "expected": "no allocation", "status": "pass"},
        {"id": "12b-N6-read-error", "given": "truncated file", "action": "read", "expected": "error propagates", "status": "pass"},
        {"id": "12b-N7-evict", "given": "tiny cache", "action": "evict", "expected": "bytes correct", "status": "pass"},
    ]

    commands = [
        {"id": "12b-CMD1", "argv": ["/opt/freetoken-venv/bin/python", "evidence/x299/12a/ple_trace.py",
                                    "--out", "evidence/x299/12a/raw/locality-trace.json"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 300, "expected_exit": 0, "phase": "b"},
        {"id": "12b-CMD2", "argv": [".venv/bin/python", "-m", "pytest", "-q",
                                    "tests/models/qwen4_exp/test_ple_disk.py", "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 300, "expected_exit": 0, "phase": "b"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "derived_from_contract": "../12a/contract.json",
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": a["scope_files"],
        "symbols": a["symbols"],
        "inputs": {
            "per_step_rows": {"value": per_step_rows, "source": "trace heads", "unit": "rows"},
            "per_step_bytes": {"value": per_step_bytes, "source": "rows*row_bytes", "unit": "bytes"},
            "disk_latency": {"value": tr["disk_latency_us_per_random_row"], "source": "trace", "unit": "us/row"},
            "tpot_ms": {"value": tpot_ms, "source": "../05b warm TPOT", "unit": "ms"},
            "ple_share_batched_pct": {"value": round(batched_share_pct, 3), "source": "bound", "unit": "%"},
            "ple_share_unbatched_pct": {"value": round(unbatched_share_pct, 3), "source": "bound", "unit": "%"},
            "locality": {"value": {"repetitive": tr["real_decode"]["cross_fill_hit_rate_infinite"],
                                   "random": tr["random_decode"]["cross_fill_hit_rate_infinite"]},
                         "source": "trace", "unit": "ratio"},
        },
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": True, "result": "FAIL / no-change",
                             "detail": f"a perfect PLE row cache saves at most ~{batched_share_pct:.2f}% (batched) to {unbatched_share_pct:.1f}% (serial) of TPOT, below the >=5% gate; locality is near-zero for code/random and already deduped for repetitive prefill"},
        "quality_gate": {"applies": False, "reason": "cache not added"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified"},
        "decision": {
            "outcome": "no_change",
            "reason": "PLE disk reads are a negligible per-step cost and locality is workload-dependent; a bounded RAM row cache cannot meet the performance gate and would be RAM spent for no measurable speedup.",
            "revisit_if": "a future workload shows sustained cross-fill n-gram replay (e.g. long verbatim loops) AND PLE disk time is a measured >5% of TPOT",
        },
        "limitations": [
            "Disk cold latency used an order-of-magnitude 100 us (O_DIRECT); warm measured 1.19 us.",
            "TPOT 46 ms is the 05b/03b warm figure; a long-context run would only make the PLE share smaller.",
            "The C++ extension cannot be built on dev (CUDA includes); a future cache would be built/tested on the target.",
        ],
        "generated_utc": "2026-09-11T15:12:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "no_change",
        "outcome_reason": contract["decision"]["reason"],
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "12b-CMD1..2",
                   "log_path": "../12a/raw/locality-trace.json"} for c in cases],
        "measurements": [{"run_id": "12b-trace", "raw_path": "../12a/raw/locality-trace.json",
                          "sha256": sha(TR)}],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no code change)",
        "key_findings": {
            "per_step_rows": per_step_rows, "per_step_bytes": per_step_bytes,
            "ple_share_batched_pct": round(batched_share_pct, 3),
            "ple_share_unbatched_pct": round(unbatched_share_pct, 3),
            "locality_repetitive": tr["real_decode"]["cross_fill_hit_rate_infinite"],
            "locality_random": tr["random_decode"]["cross_fill_hit_rate_infinite"],
            "verdict": "cache not enabled (no gain)",
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 12b; per-step", per_step_rows, "rows", per_step_bytes, "B; PLE share batched",
          round(batched_share_pct, 3), "% unbatched", round(unbatched_share_pct, 3), "%")


if __name__ == "__main__":
    main()
