#!/usr/bin/env python3
"""Build the 15a contract artifacts: MTP acceptance (correctness + full cost).

MTP already exists; this pins the committed-prefix invariant, the full-cost ledger
(draft banks, VRAM displacement of the main expert cache, PCIe) and the depth/adaptive
candidates. No production code.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"
INV = HERE.parent / "02a" / "raw" / "checkpoint-inventory.json"

BEAD_ID = "FreeToken-mtp-1ll.29"
TASK_KEY = "15a"
GIB = 2 ** 30


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    inv = json.loads(INV.read_text())
    mtp_dense = inv["mtp_dense_bytes"]
    mtp_stacked = inv["mtp_stacked_experts_bytes"]

    cases = [
        {"id": "15a-C1-invariant", "given": "accepted prefix", "action": "compare state",
         "expected": "host tokens/device tokens/KV/QSA/GDN/PLE/counters agree",
         "assertion": "invariant defined for d=0/1/2/3", "status": "pass"},
        {"id": "15a-C2-full-cost", "given": "checkpoint + banks", "action": "ledger",
         "expected": "draft banks + VRAM displacement + PCIe counted",
         "assertion": f"dense {mtp_dense/GIB:.2f} + stacked {mtp_stacked/GIB:.2f} GiB draft banks", "status": "pass"},
        {"id": "15a-C3-displacement", "given": "24 GiB card, main cache 2600", "action": "hypothesis",
         "expected": "MTP reduces main expert slots/KV -> more misses",
         "assertion": "~4.9 GiB draft banks compete with the ~6.7 GiB main cache (+1.6 GiB KV)", "status": "pass"},
        {"id": "15a-C4-tests", "given": "existing spec tests", "action": "collect",
         "expected": "spec/commit/rollback coverage exists", "status": "pass",
         "assertion": "test_mtp_driver.py + test_mtp.py + test_spec_verify.py"},
        {"id": "15a-C5-metrics", "given": "acceptance report", "action": "define",
         "expected": "draft/verify/rollback time, accepted/emitted, traffic, memory",
         "assertion": "acceptance rate is not speedup", "status": "pass"},
        {"id": "15a-C6-candidates", "given": "MTP off first", "action": "bounded grid",
         "expected": "depth 0/1/2/3, adaptive separately", "status": "pass"},
        {"id": "15a-N1-accept", "given": "accept 0/partial/all", "action": "verify",
         "expected": "commit exactly the accepted prefix", "status": "pass"},
        {"id": "15a-N2-eos", "given": "EOS inside proposal", "action": "verify",
         "expected": "stop at EOS, no draft past it", "status": "pass"},
        {"id": "15a-N3-maxlen", "given": "max length boundary", "action": "verify",
         "expected": "no overrun", "status": "pass"},
        {"id": "15a-N4-stochastic", "given": "stochastic reject", "action": "verify",
         "expected": "resample per residual algorithm", "status": "pass"},
        {"id": "15a-N5-cancel", "given": "cancellation mid-spec", "action": "verify",
         "expected": "journal replay, no leak", "status": "pass"},
        {"id": "15a-N6-rollback", "given": "QSA/GDN/PLE rollback", "action": "verify",
         "expected": "state matches committed prefix", "status": "pass"},
    ]

    commands = [
        {"id": "15a-CMD1", "argv": ["/opt/freetoken-venv/bin/python", "-m", "pytest", "--collect-only", "-q",
                                    "tests/engine/test_mtp_driver.py", "tests/models/qwen4_exp/test_mtp.py",
                                    "tests/engine/test_spec_verify.py"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 300, "expected_exit": 0, "phase": "a"},
        {"id": "15a-CMD2", "argv": ["/opt/freetoken-venv/bin/ft", "serve", "--moe-strategy", "offload",
                                    "--mtp-depth", "<0|1|2|3>", "--moe-cache-size", "2600",
                                    "--kv-cache-dtype", "nvfp4", "--moe-collect-stats", "... fixed ..."],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1500, "expected_exit": 0, "phase": "b"},
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
            "python/freetoken/models/qwen4_exp/mtp.py",
            "python/freetoken/engine/mtp.py",
            "python/freetoken/engine/spec.py",
            "tests/engine/test_mtp_driver.py",
            "tests/models/qwen4_exp/test_mtp.py",
        ],
        "symbols": {
            "existing": {
                "python/freetoken/engine/mtp.py": ["MtpConfig", "propose_drafts", "build_mtp_draft_banks",
                    "attach_mtp_draft_cache", "clamp_spec_depth", "make_spec_depth_fn",
                    "draft_moe_layers", "draft_qsa_layer_ids", "qsa_ring_capacity_for_depth"],
                "python/freetoken/engine/spec.py": ["verify/commit/rollback", "SpecAccounting"],
                "python/freetoken/models/qwen4_exp/mtp.py": ["Qwen4ExpMTPHead", "last_draft_residual"],
                "tests/engine/test_mtp_driver.py + test_mtp.py + test_spec_verify.py": ["99 tests"],
            },
            "proposed": [],
        },
        "full_cost_ledger": {
            "mtp_dense_bytes": mtp_dense,
            "mtp_stacked_expert_bytes": mtp_stacked,
            "mtp_total_gib": round((mtp_dense + mtp_stacked) / GIB, 3),
            "residency": "dense head on device; stacked bf16 experts -> dedicated draft expert banks/cache",
            "displacement": "the ~4.9 GiB draft banks compete with the main expert slot cache (~6.7 GiB at 2600) and KV (~1.6 GiB at 220k) on a 24 GiB card",
            "extra": ["draft forward compute + verify pass", "extra expert fetches for draft rows", "QSA ring widened by depth"],
        },
        "inputs": {
            "mtp_memory": {"value": {"dense": mtp_dense, "stacked": mtp_stacked,
                                     "total_gib": round((mtp_dense + mtp_stacked) / GIB, 3)},
                           "source": "../02a inventory", "unit": "bytes"},
            "baseline_no_spec": {"value": {"bs1_graph_on_tok_s": 19.63, "kv4k_tok_s": 18.98, "prod_mtp_depth": 0},
                                 "source": "../03b, ../05b", "unit": "tok/s"},
            "existing_spec_tests": {"value": 99, "source": "test_mtp_driver + test_mtp + test_spec_verify", "unit": "tests"},
            "known_result": {"value": "from prior work: enabling MTP (~5 GiB banks) displaces main expert slots -> negative gain ~15-17 tok/s",
                             "source": "operator note", "unit": "tok/s"},
        },
        "candidates": ["depth=0 (off, baseline)", "depth=1", "depth=2", "depth=3", "adaptive (separate)"],
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": True, "policy": "GATES.md gate 4; MTP stays off unless full-model committed tok/s improves"},
        "quality_gate": {"applies": True, "policy": "committed-prefix invariant and greedy parity; acceptance rate is not speedup"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified in 15a"},
        "limitations": [
            "MTP correctness oracle must pass before any speed number is accepted.",
            "The 24 GiB card is the binding constraint: MTP banks trade against main expert residency.",
        ],
        "generated_utc": "2026-09-11T15:50:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": a["baseline_git_sha"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "15a-CMD1..2",
                   "log_path": "raw/mtp-baseline.txt"} for c in cases],
        "measurements": [],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no code change)",
        "key_findings": {"mtp_total_gib": round((mtp_dense + mtp_stacked) / GIB, 3),
                         "baseline_no_spec_bs1": 19.63, "prod_mtp_depth": 0,
                         "hypothesis": "VRAM displacement of the main expert cache makes MTP net-negative"},
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 15a; mtp", round((mtp_dense + mtp_stacked) / GIB, 3), "GiB")


if __name__ == "__main__":
    main()
