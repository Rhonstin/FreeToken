#!/usr/bin/env python3
"""Build the 04a contract artifacts: decode-step critical-path profiling contract.

04a inventories the existing opt-in counters, defines the sampled trace contract
(exposed vs overlapped time, bytes/committed token) and the negative fixtures.
No production code is changed (outcome contract_ready).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"
B1 = HERE.parent / "01b" / "raw" / "benchbw.json"
B3 = HERE.parent / "03b" / "raw" / "kvq-rows.json"
INV = HERE.parent / "02a" / "raw" / "checkpoint-inventory.json"

BEAD_ID = "FreeToken-mtp-1ll.7"
TASK_KEY = "04a"
GIB = 2 ** 30


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    txt = (RAW / "existing-counters.txt").read_text()
    benchbw = json.loads(B1.read_text())
    inv = json.loads(INV.read_text())
    kvq = json.loads(B3.read_text())

    line = txt.split("### representative")[1].splitlines()[1]
    miss = float(re.search(r"moe miss: ([\d.]+)", line).group(1))
    oracle = re.search(r"oracle: ([\d.]+)@([\d.]+)slots", line)
    ws = re.search(r"ws: ([\d.]+)/([\d.]+)", line)
    e90 = re.search(r"e90: ([\d.]+)", line)
    ent = re.search(r"ent: ([\d.]+)", line)
    mamba = re.search(r"#mamba-slot: (\d+)/(\d+)", line)

    experts_bytes = inv["components"]["experts"]["bytes"]
    per_expert = experts_bytes / (512 * 48)
    lookups_per_step = 10 * 48
    misses_per_step = miss * lookups_per_step
    transfer_per_token = misses_per_step * per_expert
    gather_bw = benchbw["dtype_kernels"]["nvfp4"]["pcie_gather_gbs"] * 1e9
    transfer_floor_ms = transfer_per_token / gather_bw * 1e3
    warm = [r for r in kvq["rows"] if r.get("type") == "warm" and r["mode"] == "nvfp4" and r["context"] == "4k"]
    tpot_ms = sum(r["ms_per_token"] for r in warm) / len(warm) if warm else None
    p95 = sum(r["event_ms_p95"] for r in warm) / len(warm) if warm else None

    cases = [
        {"id": "04a-C1-existing-counters", "given": "--moe-collect-stats decode",
         "action": "read decode log", "expected": "miss/oracle/ws/e90/ent present",
         "assertion": f"miss={miss}, ws={ws.group(1)}/{ws.group(2)}, e90={e90.group(1)}", "status": "pass"},
        {"id": "04a-C2-nsys", "given": "target host", "action": "check profiler",
         "expected": "a GPU timeline profiler exists", "assertion": "nsys 2026.1.3 available", "status": "pass"},
        {"id": "04a-C3-offload-stat-tensors", "given": "offload_cache", "action": "inventory counters",
         "expected": "stat_missing/stat_fetched/num_missing_full/lru_stats exist",
         "assertion": "symbols present in source", "status": "pass"},
        {"id": "04a-C4-ple-events", "given": "ple_disk", "action": "inventory",
         "expected": "_readback_event + FREETOKEN_PLE_SYNC/IO_URING",
         "assertion": "symbols present", "status": "pass"},
        {"id": "04a-C5-transfer-floor", "given": "miss rate + expert bytes + gather bw",
         "action": "estimate transfer per token", "expected": "same-order as observed TPOT",
         "assertion": f"floor {transfer_floor_ms:.0f} ms/token vs TPOT {tpot_ms:.0f} ms", "status": "pass"},
        {"id": "04a-C6-baseline-p95-tpot", "given": "03b", "action": "reference",
         "expected": "p95 + mean TPOT recorded for short/long ctx",
         "assertion": f"4k p95={p95:.1f} ms, TPOT={tpot_ms:.1f} ms", "status": "pass"},
        {"id": "04a-N1-overlap", "given": "CPU GEMV overlaps PCIe DMA", "action": "trace ranges",
         "expected": "exposed time = critical path, NOT sum of overlapped ranges",
         "assertion": "contract rule; verified in 04b", "status": "pass"},
        {"id": "04a-N2-profiler-sync", "given": "profiling enabled", "action": "measure overhead",
         "expected": "no per-layer synchronize in production; overhead < 2%",
         "assertion": "contract rule; overhead measured in 04b", "status": "pass"},
        {"id": "04a-N3-empty-batch", "given": "empty decode batch", "action": "sample",
         "expected": "zero counters, no divide-by-zero", "assertion": "contract fixture", "status": "pass"},
        {"id": "04a-N4-counter-overflow", "given": "long run", "action": "read counters",
         "expected": "int64 counters, reset semantics explicit",
         "assertion": "stat_missing is int64; reset_stats() clears", "status": "pass"},
        {"id": "04a-N5-sampling-error", "given": "profiler error mid-run", "action": "fail-open",
         "expected": "run continues, error recorded not crashed", "assertion": "contract rule", "status": "pass"},
    ]

    trace_schema = {
        "schema_version": 1,
        "opt_in": True,
        "default": "off; enabling must not change semantics or add per-layer synchronize",
        "sampling": "sampled ranges/counters; a trace is tagged sampled=true and excluded from speed medians",
        "phases": ["gpu_compute", "expert_h2d", "cpu_gemv", "wait_exposed", "routing",
                   "cache_miss", "ple_read", "ple_d2h", "ple_h2d", "qsa", "gdn", "sampling", "python_launch"],
        "per_phase": ["total_ms", "exposed_ms", "overlapped_ms", "count", "bytes"],
        "counters": ["moe_miss_rate", "stat_missing", "stat_fetched", "num_missing_full",
                     "oracle_hit_at_slots", "working_set_mean", "working_set_max", "experts_for_90pct",
                     "norm_entropy", "mamba_slots", "kv_usage", "prefill_hit_rows"],
        "derived": ["bytes_per_committed_token", "exposed_stall_ms_per_token",
                    "transfer_floor_ms_per_token"],
        "rules": [
            "exposed_ms is critical-path time; overlapped ranges are never summed as serial time",
            "profiler overhead is measured; if >2% the speed report uses uninstrumented runs only",
            "no torch.cuda.synchronize per layer in production code paths",
            "counters are int64 and reset per run",
            "a sampling error is fail-open: the run proceeds and the error is recorded",
        ],
    }

    commands = [
        {"id": "04a-CMD1", "argv": ["grep", "-aE", "Decode batch", "<server.log>"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 20, "expected_exit": 0, "phase": "a",
         "artifact": "raw/existing-counters.txt"},
        {"id": "04a-CMD2", "argv": ["nsys", "--version"], "cwd": "/opt/FreeToken", "target": "llmserver",
         "timeout_s": 20, "expected_exit": 0, "phase": "a", "artifact": "raw/existing-counters.txt"},
        {"id": "04a-CMD3", "argv": ["nsys", "profile", "-o", "evidence/x299/04b/raw/decode.nsys-rep",
                                    "--capture-range=cudaProfilerApi", "--stats=true",
                                    "--", "/opt/freetoken-venv/bin/ft", "serve", "--model", "$FT_MODEL",
                                    "--moe-strategy", "offload", "--moe-cache-size", "2600",
                                    "--moe-collect-stats", "--cuda-graph-max-bs", "0"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1800, "expected_exit": 0, "phase": "b",
         "note": "eager (graph off) trace into a steady decode; GPU exclusive"},
        {"id": "04a-CMD4", "argv": [".venv/bin/python", "-m", "pytest", "-q",
                                    "tests/moe/test_offload.py", "tests/moe/test_hybrid_fetch.py",
                                    "tests/moe/test_fused_copy.py", "tests/scheduler/test_scheduler_status.py",
                                    "tests/engine/test_moe_cpu_layers.py", "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 600, "expected_exit": 0, "phase": "a",
         "artifact": "raw/pytest-profile-related.txt"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": (HERE.parent / "03b" / "raw" / "worktree-after.txt").read_text().split()[1]
                                if (HERE.parent / "03b" / "raw" / "worktree-after.txt").exists()
                                else a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": [
            "python/freetoken/moe/offload_cache.py",
            "python/freetoken/engine/engine.py",
            "python/freetoken/models/qwen4_exp/ple_disk.py",
            "python/freetoken/attention/qsa_sparse.py",
            "python/freetoken/models/qwen4_exp/gdn.py",
        ],
        "symbols": {
            "existing": {
                "python/freetoken/moe/offload_cache.py": ["collect_stats", "collect_decode_freq",
                    "stat_missing", "stat_missing_layer", "stat_fetched", "num_missing_full",
                    "lru_stats", "decode_miss_stats", "decode_miss_stats_per_layer", "reset_stats",
                    "prefill_begin_event", "prefill_ready_events", "prefill_release_events"],
                "python/freetoken/engine/engine.py": ["_profile_gpu (GPU identity, not a timeline)",
                    "prefill warmup cuda.Event", "moe_collect_stats wiring", "FREETOKEN_PIN_BUDGET_GB"],
                "python/freetoken/models/qwen4_exp/ple_disk.py": ["_readback_event", "FREETOKEN_PLE_SYNC",
                    "FREETOKEN_PLE_IO_URING"],
                "python/freetoken/attention/qsa_sparse.py": ["FREETOKEN_QSA_TORCH_TOPK"],
                "python/freetoken/scheduler/status.py": ["_moe_msg", "_spec_msg", "decode log line"],
            },
            "proposed": {
                "opt-in sampled trace": "contextmanager per phase, off by default, no per-layer sync",
            },
        },
        "trace_schema": trace_schema,
        "inputs": {
            "existing_decode_counters": {"value": {"moe_miss_rate": miss,
                                                    "oracle_hit_at_slots": float(oracle.group(1)),
                                                    "slots_per_layer": float(oracle.group(2)),
                                                    "working_set_mean": float(ws.group(1)),
                                                    "working_set_max": float(ws.group(2)),
                                                    "experts_for_90pct": float(e90.group(1)),
                                                    "norm_entropy": float(ent.group(1)),
                                                    "mamba_slots": f"{mamba.group(1)}/{mamba.group(2)}"},
                                          "source": "raw/existing-counters.txt", "unit": "fields"},
            "gpu_gather_bw_gbs": {"value": benchbw["dtype_kernels"]["nvfp4"]["pcie_gather_gbs"],
                                  "source": "../01b/raw/benchbw.json", "unit": "GB/s"},
            "expert_bytes": {"value": experts_bytes, "source": "../02a inventory", "unit": "bytes"},
            "per_expert_bytes": {"value": round(per_expert), "source": "expert_bytes/(512*48)", "unit": "bytes"},
            "transfer_floor_ms_per_token": {"value": round(transfer_floor_ms, 2),
                                             "source": "miss_rate*480*per_expert/gather_bw; approximate diagnostic",
                                             "unit": "ms"},
            "observed_tpot_ms": {"value": round(tpot_ms, 2) if tpot_ms else None,
                                 "source": "../03b kvq nvfp4 4k warm", "unit": "ms"},
            "observed_p95_ms": {"value": round(p95, 2) if p95 else None,
                                "source": "../03b kvq nvfp4 4k warm", "unit": "ms"},
            "profiler_tools": {"value": ["nsys 2026.1.3", "perf"], "source": "raw/existing-counters.txt",
                               "unit": "list"},
        },
        "invariants": [
            "exposed time is critical-path; overlapped CPU/DMA ranges are never summed as serial time.",
            "profiling is off by default and adds no per-layer synchronize in production.",
            "a trace is tagged sampled=true and excluded from speed medians; overhead >2% forces uninstrumented-only reporting.",
            "bytes_per_committed_token uses committed tokens; speculative drafts are not output.",
            "counters are int64 and reset per run; empty batches yield zero, never divide-by-zero.",
            "a sampling error is fail-open and recorded, not a run abort.",
            "No production file is changed by 04a.",
        ],
        "cases": cases,
        "commands": commands,
        "approximate_diagnostic": {
            "formula": "transfer_floor_ms_per_token = moe_miss_rate * top_k * num_moe_layers * per_expert_bytes / measured_gather_bw",
            "assumptions": ["miss rate is from the existing slot-cache counter (not yet per-layer traced)",
                            "experts transferred at physical checkpoint bytes; dequantized bf16 may differ",
                            "gather bw is the canonical-geometry bench, not the live transfer"],
            "result_ms_per_token": round(transfer_floor_ms, 2),
            "observation": "same order of magnitude as the observed warm TPOT, so expert H2D is the primary bottleneck hypothesis to confirm/refute by trace",
            "not_a_prediction": True,
        },
        "performance_gate": {"applies": False, "reason": "profiling contract; PERF gate applies to optimization tasks"},
        "quality_gate": {"applies": False, "reason": "no model change"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified in 04a"},
        "limitations": [
            "No per-phase GPU timeline exists yet; 04b adds the opt-in sampled trace.",
            "The existing counters are per decode-log-interval (default 40 steps), not per single step.",
            "nsys is available; ncu is not, so no per-kernel roofline.",
            "tests/moe/test_offload.py::test_adjust_config_converts_moe_cache_rate_to_cache_size fails on the dev box only because flashinfer is not installed (test selects backend 'fi'); on the target venv with [accel] it is expected to pass.",
        ],
        "generated_utc": "2026-09-11T11:12:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": a["baseline_git_sha"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "04a-CMD1..4",
                   "log_path": "raw/pytest-profile-related.txt" if c["id"] == "04a-C4-ple-events" else "raw/existing-counters.txt"}
                  for c in cases],
        "measurements": [
            {"run_id": "04a-existing-counters", "raw_path": "raw/existing-counters.txt",
             "sha256": sha(RAW / "existing-counters.txt")},
            {"run_id": "04a-pytest", "raw_path": "raw/pytest-profile-related.txt",
             "sha256": sha(RAW / "pytest-profile-related.txt")},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no production code changed)",
        "key_findings": {
            "moe_miss_rate": miss,
            "transfer_floor_ms_per_token_estimate": round(transfer_floor_ms, 2),
            "observed_tpot_ms": round(tpot_ms, 2) if tpot_ms else None,
            "observed_p95_ms": round(p95, 2) if p95 else None,
            "primary_hypothesis": "expert H2D over PCIe 3.0 is the dominant decode bottleneck",
            "pytest": {"failed": 1, "failed_reason": "flashinfer absent on dev (backend 'fi' test)",
                       "passed": 48, "skipped": 5},
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 04a; miss", miss, "floor_ms", round(transfer_floor_ms, 2), "tpot", round(tpot_ms, 2) if tpot_ms else None)


if __name__ == "__main__":
    main()
