#!/usr/bin/env python3
"""Build the 08b contract/commands/result artifacts.

Implemented the schema-5 versioned hardware fingerprint and the pure compatibility
matcher in the production reader (bench_profile.py), the profile writer
(benchbw.py) and the tests. Verified end-to-end on the target.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "08a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.16"
TASK_KEY = "08b"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    v5 = (RAW / "profile-v5.txt").read_text()
    auto = (RAW / "auto-resolution.txt").read_text()
    tamper = (RAW / "tamper-check.txt").read_text()
    matcher = (RAW / "target-matcher-check.txt").read_text()

    cases = [
        {"id": "08b-C1-schema5-written", "given": "benchbw on target", "action": "write profile",
         "expected": "schema_version 5 + fingerprint", "status": "pass",
         "assertion": "profile-v5.txt shows schema_version 5 + fingerprint"},
        {"id": "08b-C2-auto-uses-profile", "given": "serve --moe-strategy auto", "action": "resolve",
         "expected": "hybrid from the v5 profile", "status": "pass",
         "assertion": "Resolved config: moe_strategy='hybrid'"},
        {"id": "08b-C3-pure-matcher", "given": "read_machine_fingerprint/profile_compatible",
         "action": "unit tests", "expected": "pure, CPU-testable", "status": "pass",
         "assertion": "tests/moe/test_bench_profile.py 10 passed"},
        {"id": "08b-C4-no-secrets", "given": "fingerprint", "action": "inspect", "expected": "no host/serial",
         "status": "pass", "assertion": "fingerprint excludes hostname/serials"},
        {"id": "08b-C5-overlap-kept", "given": "existing overlap", "action": "reuse", "expected": "not duplicated",
         "status": "pass", "assertion": "fetch fraction 0.280 from overlapped pair"},
        {"id": "08b-C6-atomic-legacy", "given": "write/read", "action": "atomic + schema gate",
         "expected": "partial never used; legacy rejected", "status": "pass",
         "assertion": "_atomic_write_json kept; v4/None rejected"},
        {"id": "08b-N1-same-gpu-diff-cpu", "given": "tampered cpu.model", "action": "live reader",
         "expected": "reject -> offload fallback", "status": "pass",
         "assertion": "tamper-check: (None, None), reason logged"},
        {"id": "08b-N2-changed-threads", "given": "threads 4 vs 6", "action": "unit test",
         "expected": "reject", "status": "pass", "assertion": "test_changed_threads_rejected"},
        {"id": "08b-N3-same-name-diff-bdf", "given": "other BDF", "action": "live reader",
         "expected": "reject", "status": "pass", "assertion": "target-matcher-check: diff-bdf None"},
        {"id": "08b-N4-missing-fields", "given": "missing cpu.model", "action": "unit test",
         "expected": "unverified reject", "status": "pass", "assertion": "test_missing_required_field_is_unverified"},
        {"id": "08b-N5-unknown-schema", "given": "schema 4/None/'5'", "action": "unit + live",
         "expected": "reject; legacy informational", "status": "pass",
         "assertion": "target legacy reject; test_unknown_schema_rejected"},
        {"id": "08b-N6-nan-bandwidth", "given": "NaN/Inf/<=0", "action": "unit test",
         "expected": "reject", "status": "pass", "assertion": "test_bandwidth_sanity"},
    ]

    commands = [
        {"id": "08b-CMD1", "argv": [".venv/bin/python", "-m", "pytest", "-q",
                                    "tests/moe/test_bench_profile.py", "tests/moe/test_hybrid_fetch.py", "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 300, "expected_exit": 0, "phase": "b",
         "artifact": "raw/target-matcher-check.txt"},
        {"id": "08b-CMD2", "argv": ["env", "CUDA_HOME=...", "/opt/freetoken-venv/bin/ft", "bench", "bw",
                                    "--dtype", "nvfp4", "--cpu-threads", "6"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 900, "expected_exit": 0, "phase": "b",
         "artifact": "raw/profile-v5.txt"},
        {"id": "08b-CMD3", "argv": ["/opt/freetoken-venv/bin/ft", "serve", "--moe-strategy", "auto", "... fixed ..."],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 300, "expected_exit": 0, "phase": "b",
         "artifact": "raw/auto-resolution.txt"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "derived_from_contract": "../08a/contract.json",
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "changed_files": [
            "python/freetoken/moe/bench_profile.py (schema-5 gate + fingerprint + matcher + bw sanity)",
            "python/freetoken/moe/benchbw.py (write schema_version + fingerprint)",
            "tests/moe/test_hybrid_fetch.py (profiles updated to v5 + current_fingerprint)",
            "tests/moe/test_bench_profile.py (new matcher tests)",
        ],
        "scope_files": a["scope_files"],
        "symbols": {
            "existing": a["symbols"]["existing"],
            "added": {
                "python/freetoken/moe/bench_profile.py": ["read_machine_fingerprint", "profile_compatible",
                                                          "profile_bandwidths_valid", "_FP_SCHEMA",
                                                          "_cpuinfo", "_isa_tier", "_physical_cores",
                                                          "_mem_total_bytes", "_dmi_memory", "_pcie_link"],
            },
        },
        "inputs": {
            "written_profile": {"value": v5.strip(), "source": "raw/profile-v5.txt", "unit": "text"},
            "auto_resolution": {"value": auto.strip(), "source": "raw/auto-resolution.txt", "unit": "text"},
            "live_matcher": {"value": matcher.strip(), "source": "raw/target-matcher-check.txt", "unit": "text"},
            "tamper": {"value": tamper.strip(), "source": "raw/tamper-check.txt", "unit": "text"},
        },
        "invariants": [
            "The reader accepts a profile only with schema_version==5 and a fingerprint matching the current CPU/RAM/PCIe/threads/GPU.",
            "Legacy (v<5) profiles are rejected and logged as informational.",
            "The fingerprint excludes hostname, serials and MAC; RAM speed/channels are read best-effort via dmidecode (null when not permitted).",
            "A rejection logs the exact mismatching field; the engine falls back to offload.",
            "The matcher and fingerprint builder are torch-free and CPU-testable.",
        ],
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": False, "reason": "validity task; no speed claim"},
        "quality_gate": {"applies": True, "result": "PASS",
                         "detail": "unit tests + live accept/reject on the target"},
        "rollback_recipe": {
            "applies": True,
            "recipe": "restore python/freetoken/moe/bench_profile.py and benchbw.py from /opt/FreeToken/.x299-backup/; no service default change is required",
            "verified": "backup present",
        },
        "limitations": [
            "gpu.sm is caller-supplied (torch): the reader cannot derive it, so it is compared only when both sides have it.",
            "RAM speed/channels need dmidecode (root or sudo -n); when unreadable they are null and skipped.",
            "Upstream refs #278/#37/#38/#39 still not fetched (offline).",
            "The previous v4 profile at the default cache path is intentionally rejected; a fresh benchbw run replaces it with v5.",
        ],
        "generated_utc": "2026-09-11T14:30:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "implemented",
        "outcome_reason": "Versioned fingerprint + pure matcher implemented in production code and verified end-to-end on the target; prod default unchanged (config-only auto path).",
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "08b-CMD1..3",
                   "log_path": "raw/profile-v5.txt"} for c in cases],
        "measurements": [
            {"run_id": "08b-profile-v5", "raw_path": "raw/profile-v5.txt", "sha256": sha(RAW / "profile-v5.txt")},
            {"run_id": "08b-auto", "raw_path": "raw/auto-resolution.txt", "sha256": sha(RAW / "auto-resolution.txt")},
            {"run_id": "08b-matcher", "raw_path": "raw/target-matcher-check.txt", "sha256": sha(RAW / "target-matcher-check.txt")},
            {"run_id": "08b-tamper", "raw_path": "raw/tamper-check.txt", "sha256": sha(RAW / "tamper-check.txt")},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "backup at /opt/FreeToken/.x299-backup/; not rolled back (change kept)",
        "key_findings": {
            "schema_version": 5,
            "auto_resolved": "hybrid",
            "fetch_fraction": 0.2804,
            "live_rejections": ["diff-cpu", "diff-bdf", "legacy"],
            "pytest": {"passed": 13, "skipped": 2, "note": "2 CUDA-only skips on dev"},
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 08b; auto", auto.strip().splitlines()[0])


if __name__ == "__main__":
    main()
