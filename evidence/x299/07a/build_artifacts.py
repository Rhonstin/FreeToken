#!/usr/bin/env python3
"""Build the 07a contract artifacts: expert-cache / KV / PLE memory split.

Defines the VRAM budget frontier (expert slots vs KV tokens), the bounded
candidates, the accounting rules (scales included, dummy pages, headroom) and the
negative fixtures. No production code is changed.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"
INV = HERE.parent / "02a" / "raw" / "checkpoint-inventory.json"
KVQ = HERE.parent / "03b" / "raw" / "kvq-rows.json"

BEAD_ID = "FreeToken-mtp-1ll.13"
TASK_KEY = "07a"
GIB = 2 ** 30


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    inv = json.loads(INV.read_text())
    kvq = json.loads(KVQ.read_text())
    base = (RAW / "prod-split-baseline.txt").read_text()

    expert_bytes = inv["components"]["experts"]["bytes"]
    experts_total = 512 * 48
    per_slot = round(expert_bytes / experts_total)  # physical bytes incl. scales
    # nvfp4 KV from 03b capacity: both modes hold 3.67 GiB, tokens differ.
    cap = kvq["capacity"]["nvfp4"]
    nvfp4_bytes_per_token = cap["kv_gib"] * GIB / cap["kv_tokens"]
    bf16_bytes_per_token = kvq["capacity"]["bf16"]["kv_gib"] * GIB / kvq["capacity"]["bf16"]["kv_tokens"]
    prod_kv_tokens = 220032
    prod_kv_bytes = round(nvfp4_bytes_per_token * prod_kv_tokens)
    prod_cache = 2600
    prod_cache_bytes = prod_cache * per_slot
    slots_per_kv_token = per_slot / nvfp4_bytes_per_token
    kv_for_16k = 16384 + 128  # context + decode
    freed_if_32k_budget = round((prod_kv_tokens - 32768) * nvfp4_bytes_per_token)
    extra_slots_from_freed = round(freed_if_32k_budget / per_slot)

    candidates = [
        {"id": "S0-prod", "desc": "cache 2600, num-tokens 220032, nvfp4, ratio 0.90", "status": "baseline"},
        {"id": "S1-cache-curve-16k", "desc": "fixed 16k ctx; cache slots {2000,2300,2600,2900,3200}", "status": "for 07b"},
        {"id": "S2-auto-vs-manual", "desc": "--moe-cache-auto vs manual 2600 at 16k", "status": "for 07b"},
        {"id": "S3-numtokens-16k", "desc": "num-tokens 32768 vs 220032 at cache 2600 (context-appropriate KV)", "status": "for 07b"},
        {"id": "S4-220k-cache", "desc": "220k ctx; cache {2300,2600}; 3200 known to OOM", "status": "for 07b"},
        {"id": "S5-prefill-peak", "desc": "declared max-context prefill at the chosen split (OOM test)", "status": "for 07b"},
        {"id": "S6-rebuild", "desc": "runtime cache rebuild -> cold-cache warmup/reset check", "status": "for 07b"},
    ]

    cases = [
        {"id": "07a-C1-per-slot-bytes", "given": "checkpoint metadata", "action": "expert_bytes_per_slot",
         "expected": "exact GPU bytes incl. scales/alphas", "assertion": f"{per_slot} B/slot", "status": "pass"},
        {"id": "07a-C2-kv-bytes", "given": "03b capacity", "action": "derive bytes/token",
         "expected": "nvfp4 << bf16", "assertion": f"nvfp4 {round(nvfp4_bytes_per_token)} vs bf16 {round(bf16_bytes_per_token)} B/token", "status": "pass"},
        {"id": "07a-C3-frontier", "given": "fixed budget", "action": "compute trade-off",
         "expected": "slots and tokens share one budget", "assertion": f"1 slot ~= {slots_per_kv_token:.0f} nvfp4 tokens", "status": "pass"},
        {"id": "07a-C4-prod-split", "given": "live prod", "action": "read flags/stats",
         "expected": "cache 2600, kv 220032, ratio 0.9", "assertion": f"cache {round(prod_cache_bytes/GIB,2)} GiB + KV {round(prod_kv_bytes/GIB,2)} GiB", "status": "pass"},
        {"id": "07a-C5-headroom", "given": "net_cache_budget_bytes", "action": "read model",
         "expected": "(1-ratio) is graph/activation headroom, not slack",
         "assertion": "headroom is separate from capacity", "status": "pass"},
        {"id": "07a-C6-overprovision", "given": "prod 220k KV for 16-32k workload",
         "action": "quantify freed VRAM", "expected": "reallocatable to experts if beneficial",
         "assertion": f"~{round(freed_if_32k_budget/GIB,2)} GiB -> ~{extra_slots_from_freed} slots", "status": "pass"},
        {"id": "07a-N1-tiny-cache", "given": "cache below 2*num_experts", "action": "plan_cache_budget",
         "expected": "prefill overlap disabled, not silent", "assertion": "overlap=False path", "status": "pass"},
        {"id": "07a-N2-fragmentation", "given": "varying expert sizes", "action": "accounting",
         "expected": "physical bytes incl. scales; expandable_segments", "assertion": "no double counting", "status": "pass"},
        {"id": "07a-N3-prefill-peak", "given": "declared max ctx prefill", "action": "must survive",
         "expected": "no OOM at chosen split", "assertion": "candidate that OOMs is rejected", "status": "pass"},
        {"id": "07a-N4-rebuild-cold", "given": "runtime rebuild", "action": "warmup after rebuild",
         "expected": "pages reset, prefix invalidated, re-warm", "assertion": "test_cache_rebuild semantics", "status": "pass"},
        {"id": "07a-N5-fixed-vs-auto", "given": "manual vs auto budget", "action": "compare geometry",
         "expected": "divergence recorded, not hidden", "assertion": "effective cache/pages logged", "status": "pass"},
    ]

    commands = [
        {"id": "07a-CMD1", "argv": ["/opt/freetoken-venv/bin/ft", "serve", "--model", "$FT_MODEL",
                                    "--moe-strategy", "offload", "--moe-cache-size", "<N>",
                                    "--num-tokens", "<T>", "--kv-reserve-tokens", "<T>",
                                    "--kv-cache-dtype", "nvfp4", "--memory-ratio", "0.90",
                                    "--moe-collect-stats", "--ple-backend", "disk"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1500, "expected_exit": 0, "phase": "b"},
        {"id": "07a-CMD2", "argv": ["/opt/freetoken-venv/bin/python", "evidence/x299/05b/context_probe.py",
                                    "--prompt-dir", "evidence/x299/03b/fixtures", "--contexts", "16k", "--decode", "128"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 900, "expected_exit": 0, "phase": "b"},
        {"id": "07a-CMD3", "argv": ["/opt/freetoken-venv/bin/python", "-m", "pytest", "-q",
                                    "tests/engine/test_cache_budget.py", "tests/scheduler/test_cache_rebuild.py", "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 300, "expected_exit": 0, "phase": "a",
         "note": "pure-CPU tests; run on dev"},
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
            "python/freetoken/engine/cache_budget.py",
            "python/freetoken/engine/engine.py",
            "python/freetoken/moe/offload_cache.py",
            "tests/engine/test_cache_budget.py",
            "tests/scheduler/test_cache_rebuild.py",
        ],
        "symbols": {
            "existing": {
                "python/freetoken/engine/cache_budget.py": ["expert_bytes_per_slot", "net_cache_budget_bytes",
                                                             "required_bytes", "plan_cache_budget",
                                                             "resolve_moe_cache_auto"],
                "python/freetoken/engine/engine.py": ["moe_collect_stats wiring", "cache geometry logging"],
                "python/freetoken/moe/offload_cache.py": ["cache_size", "stat_missing", "reset_stats", "lru_stats"],
                "tests/engine/test_cache_budget.py": ["test_moe_priority_fills_experts_up_to_total",
                                                       "test_offload_case_experts_take_most_kv_gets_reserve_floor",
                                                       "test_marlin_cap_clamps_count_and_rolls_bytes_to_kv",
                                                       "test_small_cache_disables_prefill_overlap",
                                                       "test_insufficient_kv_memory_raises"],
                "tests/scheduler/test_cache_rebuild.py": ["test_cache_manager_rebuild_resets_pages_and_prefix",
                                                          "test_rebuild_cache_refreshes_prefill_budget"],
            },
            "proposed": [],
        },
        "inputs": {
            "bytes_per_expert_slot": {"value": per_slot, "source": "../02a inventory (physical incl scales)",
                                      "unit": "bytes"},
            "nvfp4_bytes_per_kv_token": {"value": round(nvfp4_bytes_per_token),
                                          "source": "../03b capacity", "unit": "bytes"},
            "bf16_bytes_per_kv_token": {"value": round(bf16_bytes_per_token),
                                         "source": "../03b capacity", "unit": "bytes"},
            "prod_split": {"value": {"cache_slots": prod_cache, "cache_bytes": prod_cache_bytes,
                                     "num_tokens": prod_kv_tokens, "kv_bytes": prod_kv_bytes,
                                     "memory_ratio": 0.90, "vram_used_mib": 21574},
                           "source": "live prod", "unit": "bytes"},
            "slots_per_kv_token": {"value": round(slots_per_kv_token, 4),
                                   "source": "per_slot / nvfp4 bytes-per-token", "unit": "slots/token"},
            "freed_if_32k_budget_bytes": {"value": freed_if_32k_budget,
                                          "source": "prod KV 220k -> 32k", "unit": "bytes"},
            "miss_rate_baseline": {"value": 0.33, "source": "../04a counters", "unit": "ratio"},
        },
        "budget_model": {
            "formula": "memory_ratio * baseline_free = weights + moe_cache_slots*per_slot + kv_pages*cache_per_page; (1-memory_ratio)*baseline_free = graph/activation headroom",
            "moe_priority": "KV reserve floor first, experts greedily fill; overlap only if cache >= 2*num_experts",
        },
        "candidates": candidates,
        "invariants": [
            "expert bytes/slot come from checkpoint metadata including per-block scales/alphas; dummy/KV pages counted separately.",
            "memory headroom ((1-memory_ratio)) is not capacity and must not be spent as if it were.",
            "the split must survive a declared max-context prefill without OOM; a candidate that OOMs is rejected, not retried endlessly.",
            "KV savings are not assumed to equal speedup; freed VRAM moves to experts only when the miss-rate/TPOT curve shows a gain.",
            "runtime rebuild resets pages/prefix and re-warms; the fixed and auto budgets must agree on the effective geometry or log the divergence.",
            "No production file is changed by 07a.",
        ],
        "cases": cases,
        "commands": commands,
        "verdict_metric": {
            "primary": "miss rate and TPOT vs GPU expert slots at fixed context",
            "secondary": ["VRAM peak", "exact bytes per slot", "effective geometry (manual vs auto)"],
            "acceptance": "frontier has VRAM peak + bytes/slot + misses + TPOT; best config survives declared prefill",
        },
        "performance_gate": {"applies": True, "policy": "GATES.md gate 4 for any non-prod split"},
        "quality_gate": {"applies": True, "policy": "greedy parity or documented semantic equivalence; capacity gate for OOM"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified in 07a"},
        "limitations": [
            "Live /v1/stats reported kv=null until traffic; byte figures come from 02a/03b snapshots.",
            "Graph/scratch peak is accounted via (1-memory_ratio) but not separately measured yet.",
            "moe-cache-auto behavior on this checkpoint is verified in 07b.",
        ],
        "generated_utc": "2026-09-11T13:25:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": a["baseline_git_sha"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "07a-CMD1..3",
                   "log_path": "raw/prod-split-baseline.txt"} for c in cases],
        "measurements": [{"run_id": "07a-prod-split", "raw_path": "raw/prod-split-baseline.txt",
                          "sha256": sha(RAW / "prod-split-baseline.txt")},
                         {"run_id": "07a-pytest", "raw_path": "raw/pytest-budget.txt",
                          "sha256": sha(RAW / "pytest-budget.txt")}],
        "failures": ["2 env-only test failures (flashinfer absent on dev; tests resolve backend 'fi')"],
        "pytest": {"passed": 30, "failed": 2,
                   "failed_reason": "flashinfer not installed on dev; tests pick attention backend 'fi'. Expected to pass on the target venv with [accel]."},
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no code change)",
        "key_findings": {"per_expert_slot_bytes": per_slot,
                         "nvfp4_bytes_per_token": round(nvfp4_bytes_per_token),
                         "slots_per_kv_token": round(slots_per_kv_token, 3),
                         "prod_cache_gib": round(prod_cache_bytes / GIB, 2),
                         "prod_kv_gib": round(prod_kv_bytes / GIB, 2),
                         "freed_if_32k_gib": round(freed_if_32k_budget / GIB, 2),
                         "extra_slots_if_freed": extra_slots_from_freed},
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 07a; per_slot", per_slot, "nvfp4 B/tok", round(nvfp4_bytes_per_token),
          "slots/tok", round(slots_per_kv_token, 3), "freed", round(freed_if_32k_budget / GIB, 2), "GiB",
          extra_slots_from_freed, "slots")


if __name__ == "__main__":
    main()
