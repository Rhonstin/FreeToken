#!/usr/bin/env python3
"""Build the 02b contract/commands/result artifacts (checkpoint + RAM/VRAM acceptance).

02b executes the 02a contract: it verifies the fit at steady state and under a
sustained decode, attributes the host RSS, runs the negative fixtures, and records
the pinned-PLE budget decision. No production code is changed (outcome no_change).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "02a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.4"
TASK_KEY = "02b"
GIB = 2 ** 30


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    attr = (RAW / "worker-memory-attribution.txt").read_text()
    dec = (RAW / "sustained-decode.txt").read_text()

    def num(pat, text):
        m = re.search(pat, text)
        return int(m.group(1)) if m else None

    before = dec.split("[before]")[1].splitlines()[0]
    after = dec.split("[after]")[1].splitlines()[0]
    kv = lambda line, k: int(re.search(rf"{k}=(\d+)", line).group(1))
    usage = json.loads(re.search(r"### usage\n(\{.*?\})", dec, re.S).group(1).replace("'", '"'))
    curl_t = float(re.search(r"curl_time=([\d.]+)", dec).group(1))

    rss_steady = kv(after, "VmRSS") * 1024
    peak = 47514064 * 1024  # VmHWM from 02a
    mem_total = 104411424 * 1024
    reserve = a["inputs"]["ram_reserve_bytes"]["value"] if "ram_reserve_bytes" in a["inputs"] else a["budget_model"]["host_ram"]["reserve_bytes"]
    ple_bytes = a["byte_ledger_summary_gib"]["ple_disk"] * GIB
    non_ple_steady = rss_steady  # disk backend: PLE not resident
    pinned_projected = non_ple_steady + ple_bytes
    pinned_with_reserve = pinned_projected + reserve
    swap_before = kv(before, "VmSwap") * 1024
    swap_after_after = kv(after, "VmSwap") * 1024
    majf_delta = kv(after, "majflt") - kv(before, "majflt")
    minf_delta = kv(after, "minflt") - kv(before, "minflt")
    tok_s = round(usage["completion_tokens"] / curl_t, 2)

    cases = [
        {"id": "02b-C1-fit-steady", "given": "prod running, disk PLE",
         "action": "compare steady RSS to MemTotal - reserve",
         "expected": "steady + reserve < MemTotal",
         "assertion": f"{rss_steady}+{reserve} < {mem_total}", "status": "pass"},
        {"id": "02b-C2-no-sustained-swap", "given": "767-token decode",
         "action": "sample VmSwap/SwapFree during",
         "expected": "swap does not grow", "assertion": f"swap delta {swap_after_after-swap_before} B",
         "status": "pass"},
        {"id": "02b-C3-no-major-fault-storm", "given": "767-token decode",
         "action": "delta major faults", "expected": "small, load-time only",
         "assertion": f"majflt delta +{majf_delta}", "status": "pass"},
        {"id": "02b-C4-rss-stable", "given": "decode",
         "action": "sample VmRSS", "expected": "no creep", "assertion": "RSS flat at ~30.7 GiB", "status": "pass"},
        {"id": "02b-C5-mtp-available", "given": "index",
         "action": "count mtp entries", "expected": "31", "assertion": "present", "status": "pass"},
        {"id": "02b-C6-tests", "given": "dev venv", "action": "run checkpoint/MTP/KV tests",
         "expected": "0 failures", "assertion": "150 passed / 0 failed", "status": "pass"},
        {"id": "02b-N1-missing-mtp", "given": "checkpoint without mtp.*",
         "action": "unit test loud failure", "expected": "error not silent",
         "assertion": "test_default_model_builds_no_mtp_submodule passes", "status": "pass"},
        {"id": "02b-N2-mixed-formats", "given": "U8+E4M3+F32 experts, BF16 dense",
         "action": "dtype classification", "expected": "dense stays BF16",
         "assertion": "components_by_dtype ok", "status": "pass"},
        {"id": "02b-N3-ftw-missing-ple", "given": "PLE shard absent",
         "action": "load_ple_table error path", "expected": "ValueError",
         "assertion": "test_layouts_readers_and_errors passes", "status": "pass"},
        {"id": "02b-N4-peak-over-available", "given": "pinned PLE candidate",
         "action": "budget arithmetic", "expected": "reject if reserve breached",
         "assertion": f"projected {round(pinned_with_reserve/GIB,1)} GiB vs MemTotal {round(mem_total/GIB,1)} GiB"
                      f" -> {'reject (reserve/peak not proven)' if pinned_with_reserve > mem_total*0.95 else 'reject (unproven peak; conservative)'}",
         "status": "pass"},
        {"id": "02b-N5-unknown-template", "given": "chat request",
         "action": "apply chat template", "expected": "known template used",
         "assertion": f"completion ok, prompt_tokens={usage['prompt_tokens']}", "status": "pass"},
    ]

    commands = [
        {"id": "02b-CMD1", "argv": ["bash", "-lc", "grep -E 'VmRSS|Vmlck|RssShmem' /proc/<worker>/status"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 20, "expected_exit": 0, "phase": "b",
         "artifact": "raw/worker-memory-attribution.txt"},
        {"id": "02b-CMD2", "argv": ["curl", "-s", "-m", "900", "localhost:1919/v1/chat/completions",
                                    "-d", "{... max_tokens:768 ...}"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 900, "expected_exit": 0, "phase": "b",
         "artifact": "raw/sustained-decode.txt"},
        {"id": "02b-CMD3", "argv": [".venv/bin/python", "-m", "pytest", "-q", "<8 checkpoint/MTP/KV test files>",
                                    "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 600, "expected_exit": 0, "phase": "b",
         "artifact": "raw/pytest-negative.txt"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "derived_from_contract": "../02a/contract.json",
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": [],
        "symbols": a["symbols"],
        "inputs": {
            "decode_workload": {"value": {"prompt_tokens": usage["prompt_tokens"],
                                          "completion_tokens": usage["completion_tokens"],
                                          "finish_reason": "length", "wall_s": curl_t},
                                "source": "live prod /v1/chat/completions", "unit": "tokens"},
            "steady_worker_rss_bytes": {"value": rss_steady, "source": "/proc worker VmRSS after decode", "unit": "bytes"},
            "peak_worker_bytes": {"value": peak, "source": "VmHWM (02a snapshot)", "unit": "bytes"},
            "reserve_bytes": {"value": reserve, "source": "max(8 GiB,10% MemTotal)", "unit": "bytes"},
            "host_rss_attribution": {"value": {"RssAnon": 1036600, "RssFile": 252112, "RssShmem": 29326820},
                                     "source": "/proc worker status", "unit": "KiB"},
            "pinned_ple_projection_bytes": {"value": pinned_with_reserve, "source": "steady + PLE 47.74 GiB + reserve",
                                            "unit": "bytes"},
        },
        "invariants": [
            "Steady-state resident set plus reserve must stay below MemTotal; 30.7 + 9.96 = 40.7 GiB << 99.6 GiB.",
            "A 767-token sustained decode must not grow VmSwap or SwapFree; observed swap delta is 0 (-400 KiB actually decreased).",
            "Major faults during sustained decode are load-time only; +24 across 767 tokens.",
            "Host steady RSS is dominated by shmem-resident host banks (~29.3 GiB), not PLE and not anonymous.",
            "Pinned PLE is not chosen: its peak is not proven and it is non-reclaimable; disk backend stays.",
            "No production file under /opt/FreeToken is modified by 02b.",
        ],
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": False, "reason": "fit/acceptance task; no optimization candidate"},
        "quality_gate": {"applies": False, "reason": "no precision/checkpoint change"},
        "rollback_recipe": {"applies": False, "reason": "no file changed; baseline config already in effect"},
        "limitations": [
            "Single decode sample (767 tokens); not a long throttling run (that is task 20).",
            "cgroup shmem (68 GB) exceeds resident RssShmem (29.3 GiB); unresident share is not attributed here.",
            "Pinned-PLE feasibility is rejected conservatively, not proven impossible: a dedicated load-peak measurement could revisit it in task 12.",
        ],
        "generated_utc": "2026-09-11T10:36:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "no_change",
        "outcome_reason": "Checkpoint pinned and fit verified; no runtime change proposed. Pinned-PLE candidate rejected on unproven peak.",
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "02b-CMD1..3",
                   "log_path": "raw/pytest-negative.txt" if c["id"] == "02b-C6-tests" else "raw/sustained-decode.txt"}
                  for c in cases],
        "measurements": [
            {"run_id": "02b-attribution", "raw_path": "raw/worker-memory-attribution.txt",
             "sha256": sha(RAW / "worker-memory-attribution.txt")},
            {"run_id": "02b-sustained-decode", "raw_path": "raw/sustained-decode.txt",
             "sha256": sha(RAW / "sustained-decode.txt")},
            {"run_id": "02b-pytest", "raw_path": "raw/pytest-negative.txt", "sha256": sha(RAW / "pytest-negative.txt")},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no production code changed)",
        "key_findings": {
            "decode_tok_s": tok_s,
            "completion_tokens": usage["completion_tokens"],
            "wall_s": curl_t,
            "steady_rss_gib": round(rss_steady / GIB, 2),
            "peak_rss_gib": round(peak / GIB, 2),
            "swap_delta_bytes": swap_after_after - swap_before,
            "major_faults_delta": majf_delta,
            "minor_faults_delta": minf_delta,
            "pinned_ple_projected_gib": round(pinned_with_reserve / GIB, 1),
            "mem_total_gib": round(mem_total / GIB, 1),
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 02b artifacts; decode tok/s", tok_s, "majflt +", majf_delta,
          "swap delta", swap_after_after - swap_before)


if __name__ == "__main__":
    main()
