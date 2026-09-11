#!/usr/bin/env python3
"""Build the 15b contract/commands/result artifacts.

Decision: no_change — MTP stays disabled. The ~4.9 GiB draft banks do not fit on the
24 GiB card at the declared 220k context with a prod-realistic main cache (depth1 OOMs
at both fixed 2600 and --moe-cache-auto), and forcing a fit by shrinking the main cache
raises the miss rate (net-negative). Correctness tests pass, but correctness alone is
not acceptance; the fit gate fails first.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "15a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.30"
TASK_KEY = "15b"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def row(p, mode):
    for ln in pathlib.Path(p).read_text().splitlines():
        d = json.loads(ln)
        if d["mode"] == mode:
            return d
    return None


def main() -> None:
    a = json.loads(A.read_text())
    d0 = row(RAW / "mtp-ab.jsonl", "depth0")
    a0 = row(RAW / "mtp-auto.jsonl", "auto-depth0")

    cases = [
        {"id": "15b-C1-correctness", "given": "existing spec suite", "action": "run",
         "expected": "green", "status": "pass", "assertion": "99 collected; 97 passed, 2 skipped"},
        {"id": "15b-C2-baseline-off", "given": "prod cache 2600, depth0", "action": "measure",
         "expected": "no-spec baseline", "status": "pass",
         "assertion": f"{d0['decode_tok_s']:.2f} tok/s, miss 0.35, VRAM 20.65 GiB"},
        {"id": "15b-C3-depth1-2600", "given": "cache 2600, depth1", "action": "launch",
         "expected": "fits or OOM recorded", "status": "pass",
         "assertion": "OOM: needs 972 MiB, 344 MiB free -> draft banks displace the main cache"},
        {"id": "15b-C4-depth1-auto", "given": "--moe-cache-auto, depth1", "action": "launch",
         "expected": "fits or OOM recorded", "status": "pass",
         "assertion": "OOM at KV-pool alloc -> MTP banks are not in the auto budget"},
        {"id": "15b-C5-displacement", "given": "depth0 auto baseline", "action": "compare",
         "expected": "show the cost of making room", "status": "pass",
         "assertion": f"depth0 auto {a0['decode_tok_s']:.2f} tok/s, cache 3447, miss 0.29"},
        {"id": "15b-C6-no-enable", "given": "fit gate failed", "action": "keep MTP off",
         "expected": "no default change", "status": "pass", "assertion": "prod mtp_depth=0"},
        {"id": "15b-N1-accept", "given": "accept 0/partial/all", "action": "unit tests",
         "expected": "commit exactly accepted prefix", "status": "pass", "assertion": "spec suite green"},
        {"id": "15b-N2-eos", "given": "EOS in proposal", "action": "unit tests", "expected": "stop", "status": "pass"},
        {"id": "15b-N3-maxlen", "given": "max length", "action": "unit tests", "expected": "no overrun", "status": "pass"},
        {"id": "15b-N4-stochastic", "given": "stochastic reject", "action": "unit tests", "expected": "residual resample", "status": "pass"},
        {"id": "15b-N5-cancel", "given": "cancellation", "action": "unit tests", "expected": "journal replay", "status": "pass"},
        {"id": "15b-N6-rollback", "given": "QSA/GDN/PLE rollback", "action": "unit tests", "expected": "state matches prefix", "status": "pass"},
    ]

    commands = [
        {"id": "15b-CMD1", "argv": [".venv/bin/python", "-m", "pytest", "-q",
                                    "tests/engine/test_mtp_driver.py", "tests/models/qwen4_exp/test_mtp.py",
                                    "tests/engine/test_spec_verify.py", "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 600, "expected_exit": 0, "phase": "b",
         "artifact": "../15a/raw/mtp-baseline.txt"},
        {"id": "15b-CMD2", "argv": ["/opt/freetoken-venv/bin/ft", "serve", "--moe-strategy", "offload",
                                    "--moe-cache-size", "2600|--moe-cache-auto", "--mtp-depth", "0|1",
                                    "--kv-cache-dtype", "nvfp4", "--num-tokens", "220032", "--moe-collect-stats"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1500, "expected_exit": 0, "phase": "b"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "derived_from_contract": "../15a/contract.json",
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": a["scope_files"],
        "symbols": a["symbols"],
        "inputs": {
            "mtp_banks_gib": {"value": a["full_cost_ledger"]["mtp_total_gib"], "source": "../02a", "unit": "GiB"},
            "depth0_cache2600": {"value": {"tok_s": round(d0["decode_tok_s"], 2), "miss": 0.35, "vram_mib": 20652},
                                 "source": "raw/mtp-ab.jsonl", "unit": "tok/s"},
            "depth0_auto": {"value": {"tok_s": round(a0["decode_tok_s"], 2), "cache": 3447, "miss": 0.29, "vram_mib": 22892},
                            "source": "raw/mtp-auto.jsonl", "unit": "tok/s"},
            "depth1_results": {"value": {"cache2600": "OOM", "auto": "OOM", "cache800": "config assert (overlap needs >=1024)"},
                               "source": "raw/mtp-summary.txt", "unit": "outcome"},
            "spec_tests": {"value": "99 collected; 97 passed, 2 skipped", "source": "../15a/raw/mtp-baseline.txt", "unit": "tests"},
        },
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": True, "result": "FAIL / no-change",
                             "detail": "depth1 does not fit at the declared 220k context; the only way to fit is to shrink the main expert cache, which raises the miss rate and makes MTP net-negative"},
        "quality_gate": {"applies": True, "result": "PASS",
                         "detail": "committed-prefix invariants covered by 99 spec tests (97 pass, 2 skip); correctness is necessary but not sufficient"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified; prod mtp_depth=0 unchanged"},
        "decision": {
            "outcome": "no_change",
            "reason": "MTP's ~4.9 GiB draft banks (dense 0.17 + stacked bf16 experts 4.69) do not fit on the 24 GiB card next to the main expert cache and 220k KV: depth1 OOMs at cache 2600 and under --moe-cache-auto (auto does not budget for the draft banks). Fitting requires shrinking the main cache -> more misses -> net-negative (operator measurement ~15-17 tok/s vs ~20). MTP stays off.",
            "revisit_if": "the draft banks are shrinkable (quantized/smaller MTP experts, or a shared cache) or the context budget is reduced so both fit with the main cache unchanged",
        },
        "limitations": [
            "Not all depths were measured end-to-end: depth1 already fails the fit gate, so depth2/3 are strictly worse.",
            "The 4k probe uses the declared 220k KV pool; a smaller context could fit MTP but at a lower main-cache hit rate too.",
            "A cache-800 fit attempt hit the prefill_overlap >= 2*num_experts assert rather than an OOM.",
        ],
        "generated_utc": "2026-09-11T16:20:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "no_change",
        "outcome_reason": contract["decision"]["reason"],
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "15b-CMD1..2",
                   "log_path": "raw/mtp-summary.txt"} for c in cases],
        "measurements": [
            {"run_id": "15b-depth0", "raw_path": "raw/mtp-ab.jsonl", "sha256": sha(RAW / "mtp-ab.jsonl")},
            {"run_id": "15b-auto", "raw_path": "raw/mtp-auto.jsonl", "sha256": sha(RAW / "mtp-auto.jsonl")},
            {"run_id": "15b-summary", "raw_path": "raw/mtp-summary.txt", "sha256": sha(RAW / "mtp-summary.txt")},
        ],
        "failures": ["depth1 OOM at cache 2600 and auto (fit gate)"],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (MTP default off; nothing changed)",
        "key_findings": {
            "depth0_cache2600_tok_s": round(d0["decode_tok_s"], 2),
            "depth0_auto_tok_s": round(a0["decode_tok_s"], 2),
            "depth1": "OOM (does not fit)",
            "mtp_banks_gib": a["full_cost_ledger"]["mtp_total_gib"],
            "verdict": "MTP stays disabled (fit gate); correctness green",
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 15b; depth0@2600", round(d0["decode_tok_s"], 2), "auto", round(a0["decode_tok_s"], 2), "depth1 OOM")


if __name__ == "__main__":
    main()
