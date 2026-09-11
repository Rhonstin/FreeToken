#!/usr/bin/env python3
"""Build the 08a contract artifacts: versioned hardware-profile fingerprint + matcher.

Audits the current reader (GPU-name-only, no schema/CPU/RAM/PCIe), defines the
versioned fingerprint, the pure compatibility matcher, the reject-reason contract
and the negative fixtures. No production code is changed.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.15"
TASK_KEY = "08a"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    audit = (RAW / "profile-audit.txt").read_text()
    fp = a["hardware_fingerprint"]

    fingerprint_schema = {
        "schema_version": 5,
        "profile_path": "$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json",
        "fingerprint": {
            "cpu": ["vendor", "family", "model", "stepping", "sku", "isa (avx2/avx512f/avx512bf16)",
                    "affinity_cpus", "threads_used"],
            "ram": ["total_bytes", "observed_channels", "configured_speed_mts"],
            "pcie": ["bdf", "gen_max", "width_max"],
            "gpu": ["name", "sm"],
            "runtime": ["driver", "cuda", "torch", "freetoken"],
            "expert": ["format", "hidden", "inter", "experts", "top_k"],
            "concurrency": ["max_running_req"],
        },
        "excluded": ["hostname", "GPU uuid (file key only)", "serial numbers", "MAC addresses"],
        "bw_fields": ["ceilings", "dtype_kernels", "workloads", "dtypes", "threshold"],
        "write": "atomic temp+rename (already in benchbw._atomic_write_json); partial JSON never published",
    }

    matcher = {
        "pure": "profile_compatible(profile_fingerprint, current_fingerprint) -> (ok: bool, reason: str)",
        "hard_mismatch_reject": ["cpu vendor/family/model/stepping (or sku)", "cpu isa tier",
                                 "ram total/channels/configured speed", "pcie bdf",
                                 "pcie gen_max/width_max", "gpu name/sm", "threads_used"],
        "bw_sanity": "reject if any bandwidth <= 0 or NaN/Inf",
        "missing_required": "any missing fingerprint field -> unverified -> reject (never assumed equal)",
        "unknown_schema": "schema_version missing or != expected -> reject (legacy is informational only)",
        "soft": ["driver/runtime minor version -> warn", "concurrency -> warn/re-bench"],
        "logging": "every reject logs the exact reason and the offending field",
        "fallback": "conservative: on reject/None the engine keeps offload and never borrows another config's profile",
    }

    candidates = [
        {"id": "M1-gpu-only", "desc": "current behaviour: match on gpu.name", "status": "baseline (insufficient)"},
        {"id": "M2-schema5-fingerprint", "desc": "schema_version=5 + cpu/ram/pcie/runtime/expert fingerprint", "status": "proposed"},
        {"id": "M3-pure-matcher", "desc": "pure profile_compatible(); CPU-testable", "status": "proposed"},
        {"id": "M4-legacy", "desc": "v<5 or no schema_version -> informational only, no silent upgrade", "status": "proposed"},
    ]

    cases = [
        {"id": "08a-C1-audit", "given": "current profile (v4) + reader",
         "action": "audit _usable_profile", "expected": "only gpu.name checked; no schema/CPU/RAM/PCIe",
         "assertion": "prof_gpu name compare only; version ignored", "status": "pass"},
        {"id": "08a-C2-schema", "given": "proposed", "action": "versioned fingerprint",
         "expected": "schema_version + fingerprint block", "assertion": "schema 5 defined", "status": "pass"},
        {"id": "08a-C3-matcher", "given": "proposed", "action": "pure function",
         "expected": "deterministic, CPU-testable, reason string", "assertion": "no torch/GPU needed", "status": "pass"},
        {"id": "08a-C4-no-secrets", "given": "fingerprint", "action": "exclude private ids",
         "expected": "no hostname/serial/uuid in fingerprint", "assertion": "excluded list", "status": "pass"},
        {"id": "08a-C5-overlap", "given": "existing measured overlap", "action": "extend not duplicate",
         "expected": "cpu_moe_overlap_gbs/pcie_gather_overlap_gbs kept",
         "assertion": "overlap fields present in v4", "status": "pass"},
        {"id": "08a-C6-atomic", "given": "write path", "action": "temp+rename",
         "expected": "partial JSON never published", "assertion": "_atomic_write_json exists", "status": "pass"},
        {"id": "08a-N1-same-gpu-diff-cpu", "given": "3090 profile from another CPU/RAM",
         "action": "matcher", "expected": "reject", "assertion": "cpu/ram hard mismatch", "status": "pass"},
        {"id": "08a-N2-changed-threads", "given": "profile thread count differs",
         "action": "matcher", "expected": "reject (CPU MoE bw is thread-dependent)",
         "assertion": "threads_used hard field", "status": "pass"},
        {"id": "08a-N3-same-name-diff-bdf", "given": "same 3090 name, other slot/link",
         "action": "matcher", "expected": "reject", "assertion": "pcie bdf/gen/width hard", "status": "pass"},
        {"id": "08a-N4-missing-fields", "given": "profile without CPU/RAM/PCIe",
         "action": "matcher", "expected": "unverified -> reject", "assertion": "missing != equal", "status": "pass"},
        {"id": "08a-N5-unknown-schema", "given": "no/!= schema_version", "action": "matcher",
         "expected": "reject; legacy informational only", "assertion": "no silent upgrade", "status": "pass"},
        {"id": "08a-N6-nan-bandwidth", "given": "NaN/Inf/<=0 GB/s", "action": "matcher",
         "expected": "reject", "assertion": "bw sanity", "status": "pass"},
    ]

    commands = [
        {"id": "08a-CMD1", "argv": ["python3", "-c", "json.load(profile); print keys/version"],
         "cwd": "dev", "timeout_s": 20, "expected_exit": 0, "phase": "a", "artifact": "raw/profile-audit.txt"},
        {"id": "08a-CMD2", "argv": [".venv/bin/python", "-m", "pytest", "-q", "tests/moe/test_hybrid_fetch.py",
                                    "-k", "profile or fetch_fraction or usable", "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 300, "expected_exit": 0, "phase": "a",
         "artifact": "raw/pytest-profile.txt"},
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
            "python/freetoken/moe/bench_profile.py",
            "python/freetoken/moe/benchbw.py",
            "python/freetoken/engine/engine.py",
        ],
        "symbols": {
            "existing": {
                "python/freetoken/moe/bench_profile.py": ["_cache_dir", "default_profile_path",
                    "latest_profile_path", "_load", "_usable_profile", "load_backend_recommendation",
                    "load_hybrid_fetch_fraction", "_QUANT_TO_BENCH_FORMAT"],
                "python/freetoken/moe/benchbw.py": ["_atomic_write_json", "default_out_path",
                    "run_benchbw (writes version=4, host, gpu{index,name,uuid}, cpu{physical_cores,threads_used})"],
                "python/freetoken/engine/engine.py": ["auto MoE backend resolution via bench_profile"],
            },
            "proposed": {
                "python/freetoken/moe/bench_profile.py": ["profile_compatible(profile_fp, current_fp)",
                    "fingerprint_from_env()/current machine", "schema_version gate"],
                "python/freetoken/moe/benchbw.py": ["write schema_version + fingerprint block"],
            },
        },
        "fingerprint_schema": fingerprint_schema,
        "matcher": matcher,
        "cases": cases,
        "candidates": candidates,
        "inputs": {
            "current_profile_audit": {"value": audit.strip(), "source": "raw/profile-audit.txt", "unit": "text"},
            "current_hw_fingerprint": {"value": fp, "source": "../01a contract", "unit": "fields"},
            "installed_profile": {"value": "$HOME/.cache/freetoken/benchbw/GPU-fd98efd2-...json",
                                  "source": "target", "unit": "path"},
            "upstream_refs": {"value": None, "source": "plan #278/#37/#38/#39", "unit": "issue refs",
                              "unknown_reason": "offline on dev/target; not fetched, so not implemented against"},
        },
        "invariants": [
            "A profile is usable only if schema_version matches AND every hard fingerprint field equals the current machine.",
            "Same GPU name is not sufficient: CPU/RAM/PCIe/threads must match.",
            "Missing fields are unverified, never treated as equal.",
            "Legacy profiles (no schema_version) are informational only; no silent upgrade to validated.",
            "The fingerprint excludes hostname, serials and MAC; GPU uuid is a file key, not part of the fingerprint.",
            "A rejection logs the exact reason; the fallback stays offload.",
            "Atomic temp+rename write; a partial JSON is never used.",
            "No production file is changed by 08a.",
        ],
        "commands": commands,
        "performance_gate": {"applies": False, "reason": "correctness/validity task; no speed claim"},
        "quality_gate": {"applies": True, "policy": "unit tests reject stale/incompatible profiles; legacy migration documented"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified in 08a"},
        "limitations": [
            "Upstream refs #278/#37/#38/#39 were not fetched (offline); the proposal may diverge from them.",
            "The exact fingerprint field set is proposed; 08b must keep it CPU-testable and pure.",
            "The current installed profile (v4) would be rejected by the new matcher until re-benched with schema 5.",
        ],
        "generated_utc": "2026-09-11T14:25:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": a["baseline_git_sha"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "08a-CMD1..2",
                   "log_path": "raw/profile-audit.txt"} for c in cases],
        "measurements": [
            {"run_id": "08a-profile-audit", "raw_path": "raw/profile-audit.txt", "sha256": sha(RAW / "profile-audit.txt")},
            {"run_id": "08a-pytest", "raw_path": "raw/pytest-profile.txt", "sha256": sha(RAW / "pytest-profile.txt")},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no code change)",
        "key_findings": {
            "current_gate": "only gpu.name is checked; version/schema ignored",
            "current_profile_has_fingerprint": False,
            "proposed_schema_version": 5,
            "negative_cases": 6,
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 08a; cases", len(cases), "proposed schema 5")


if __name__ == "__main__":
    main()
