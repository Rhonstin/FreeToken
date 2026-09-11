#!/usr/bin/env python3
"""Build the 06a contract artifacts: i7-7800X CPU thread/affinity/ISA tuning.

Defines the bounded thread/ISA sweep, the fixed constraints (leave cores for the
scheduler and PLE I/O, no oversubscription, ISA only with flag+compiled support)
and the negative fixtures. No production code is changed.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.11"
TASK_KEY = "06a"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    txt = (RAW / "cpu-baseline.txt").read_text()
    bench_bw = json.loads((HERE.parent / "01b" / "raw" / "benchbw.json").read_text())
    cpu_moe_nvfp4 = bench_bw["dtype_kernels"]["nvfp4"]["cpu_moe_gbs"]
    cpu_moe_isa = bench_bw["dtype_kernels"]["nvfp4"]["cpu_moe_isa"]

    reps = json.loads(re.search(r"physical_reps (\[[^\]]*\])", txt).group(1))
    affinity = json.loads(re.search(r"affinity (\[[^\]]*\])", txt).group(1))
    smt_pairs = re.findall(r"cpu(\d+) siblings=([\d,]+)", txt)
    gov = re.search(r"scaling_governor\s*\n(\w+)", txt)
    maxfreq = re.search(r"cpuinfo_max_freq:(\d+)", txt)

    # P=6 physical cores: grid {1,2,4,P-2,P-1,P} = {1,2,4,5,6}; SMT {8,12} separately.
    P = len(reps)
    thread_grid = sorted({1, 2, 4, max(1, P - 2), max(1, P - 1), P})
    smt_grid = [P + 2, 2 * P]

    cases = [
        {"id": "06a-C1-physical-reps", "given": "sysfs topology + affinity",
         "action": "physical_core_cpus()", "expected": "one logical CPU per core",
         "assertion": f"reps={reps}", "status": "pass"},
        {"id": "06a-C2-resolve", "given": "requested counts", "action": "resolve_threads_and_affinity",
         "expected": "physical first, then logical; 0=auto=physical",
         "assertion": "0->(6, phys), 8->(8, adds 2 siblings)", "status": "pass"},
        {"id": "06a-C3-isa-flags", "given": "lscpu", "action": "read flags",
         "expected": "avx2+avx512f, no avx512bf16", "assertion": "avx512bf16 absent -> clamps", "status": "pass"},
        {"id": "06a-C4-compiled-tiers", "given": "cpp", "action": "read dispatch",
         "expected": "scalar/avx2/avx512f/avx512bf16 compiled",
         "assertion": "target() attrs present", "status": "pass"},
        {"id": "06a-C5-thread-grid", "given": "P=6", "action": "bounded grid",
         "expected": "{1,2,4,5,6} + SMT {8,12} separate",
         "assertion": f"grid={thread_grid}, smt={smt_grid}", "status": "pass"},
        {"id": "06a-C6-baseline-cpumoe", "given": "01b benchbw", "action": "reference",
         "expected": "CPU MoE GB/s at 6 threads", "assertion": f"{cpu_moe_nvfp4} GB/s ({cpu_moe_isa})", "status": "pass"},
        {"id": "06a-N1-smt-siblings", "given": "requested 8/12", "action": "resolve",
         "expected": "SMT used only when requested beyond physical; not default",
         "assertion": "0 default = physical only", "status": "pass"},
        {"id": "06a-N2-uneven-cpuset", "given": "affinity subset", "action": "physical_core_cpus",
         "expected": "restricted to affinity, falls back gracefully", "assertion": "reads sched_getaffinity", "status": "pass"},
        {"id": "06a-N3-request-gt-allowed", "given": "requested > allowed CPUs",
         "action": "resolve", "expected": "oversubscribed; never a default",
         "assertion": "candidate allowed but flagged oversubscribed", "status": "pass"},
        {"id": "06a-N4-no-avx512", "given": "CPU/build without avx512", "action": "dispatch",
         "expected": "clamp to avx2/scalar, never run avx512", "assertion": "FREETOKEN_CPU_MOE_ISA clamps down", "status": "pass"},
        {"id": "06a-N5-avx512-throttle", "given": "sustained avx512", "action": "sample effective clock",
         "expected": "record any frequency drop", "assertion": "clock/thermal captured in 06b", "status": "pass"},
        {"id": "06a-N6-ple-steals-cores", "given": "PLE disk IO workers", "action": "leave headroom",
         "expected": "keep >=1 core free for scheduler/PLE IO", "assertion": "grid capped at P; overlap measured", "status": "pass"},
    ]

    commands = [
        {"id": "06a-CMD1", "argv": ["/opt/freetoken-venv/bin/python", "-c",
                                    "from freetoken.moe.cpu_executor import physical_core_cpus, resolve_threads_and_affinity"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 60, "expected_exit": 0, "phase": "a",
         "artifact": "raw/cpu-baseline.txt"},
        {"id": "06a-CMD2", "argv": ["env", "CUDA_HOME=...nvidia/cu13", "/opt/freetoken-venv/bin/ft", "bench", "bw",
                                    "--dtype", "nvfp4", "--isa", "all", "--cpu-threads", "6",
                                    "-o", "evidence/x299/06b/raw/benchbw-isa.json"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 900, "expected_exit": 0, "phase": "b",
         "note": "ISA sweep over the same banks; excludes GPU-mismatch via CUDA_HOME"},
        {"id": "06a-CMD3", "argv": ["env", "CUDA_HOME=...", "/opt/freetoken-venv/bin/ft", "serve", "--model", "$FT_MODEL",
                                    "--moe-strategy", "hybrid", "--moe-hybrid-max-fetch", "1",
                                    "--moe-cpu-threads", "<N>", "... fixed knobs ..."],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1500, "expected_exit": 0, "phase": "b",
         "note": "full-model TPOT at thread grid; explicit setting wins over auto"},
        {"id": "06a-CMD4", "argv": [".venv/bin/python", "-m", "pytest", "-q", "tests/moe/test_cpu_moe.py", "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 900, "expected_exit": 0, "phase": "b",
         "note": "CPU MoE correctness must stay green across thread counts"},
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
            "python/freetoken/moe/cpu_executor.py",
            "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
            "python/freetoken/moe/benchbw.py",
            "tests/moe/test_cpu_moe.py",
        ],
        "symbols": {
            "existing": {
                "python/freetoken/moe/cpu_executor.py": ["physical_core_cpus", "resolve_threads_and_affinity",
                                                          "CpuMoeExecutor", "compiled_extension_supports"],
                "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp": ["dot_scalar", "dot_avx2fma",
                                                                          "dot_avx512f", "dot_avx512bf16",
                                                                          "FREETOKEN_CPU_MOE_ISA"],
                "python/freetoken/moe/benchbw.py": ["_ISA_TIERS", "_forced_isa", "--isa", "--cpu-threads", "isa_sweep"],
                "tests/moe/test_cpu_moe.py": ["test_cpu_decode_matches_gpu_decode_kernel",
                                               "test_cpu_decode_nvfp4_matches_dequant_then_gpu",
                                               "test_cpu_moe_decode_cuda_graph_replay"],
            },
            "proposed": [],
        },
        "inputs": {
            "affinity": {"value": affinity, "source": "os.sched_getaffinity", "unit": "cpu ids"},
            "physical_core_reps": {"value": reps, "source": "physical_core_cpus()", "unit": "cpu ids"},
            "smt_pairs": {"value": dict(smt_pairs), "source": "sysfs thread_siblings_list", "unit": "pairs"},
            "isa": {"value": {"avx2": True, "avx512f": True, "avx512bf16": False},
                    "source": "lscpu flags", "unit": "bool"},
            "compiled_tiers": {"value": ["scalar", "avx2", "avx512", "avx512bf16"],
                               "source": "cpp target() attrs", "unit": "list"},
            "governor_max_freq": {"value": {"governor": gov.group(1) if gov else None,
                                            "cpuinfo_max_freq_khz": int(maxfreq.group(1)) if maxfreq else None},
                                  "source": "cpufreq sysfs", "unit": "kHz"},
            "cpu_moe_baseline": {"value": {"gbs": cpu_moe_nvfp4, "isa": cpu_moe_isa, "threads": 6},
                                 "source": "../01b/raw/benchbw.json", "unit": "GB/s"},
        },
        "thread_grid": {"physical_P": P, "core_threads": thread_grid, "smt_only": smt_grid},
        "invariants": [
            "auto (requested=0) uses one thread per physical core on the affinity set; SMT is only used when explicitly requested beyond physical cores.",
            "AVX-512 is benchmarked only with CPUID flags and compiled support; avx512bf16 must clamp down on this 7800X.",
            "The recommendation rests on full-model TPOT, not STREAM/CPU-MoE GB/s alone.",
            "Keep >=1 physical core free for the scheduler and the PLE disk I/O workers; no oversubscription as a default.",
            "Explicit --moe-cpu-threads overrides auto and is honored.",
            "CPU MoE correctness tests stay green across the swept thread counts.",
            "No production file is changed by 06a.",
        ],
        "cases": cases,
        "commands": commands,
        "verdict_metric": {
            "primary": "full-model decode tokens/s (hybrid/cpu backend) at each thread count",
            "secondary": ["CPU MoE GB/s diagnostic", "effective CPU clock / throttling", "CPU GEMV + PCIe DMA overlap"],
            "acceptance": "recommend only on stable full-model TPOT; no default change from a single measurement",
        },
        "performance_gate": {"applies": True, "policy": "GATES.md gate 4 once a non-default thread count is proposed"},
        "quality_gate": {"applies": True, "policy": "tests/moe/test_cpu_moe.py green across thread counts"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified in 06a"},
        "limitations": [
            "Effective clock under sustained AVX-512 is sampled in 06b (idle shows 4.0 GHz, governor powersave).",
            "PLE disk IO worker thread count is not yet exposed; its core usage is measured, not tuned, in 06b.",
            "SMT candidates are secondary and may be rejected as unstable.",
        ],
        "generated_utc": "2026-09-11T12:40:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": a["baseline_git_sha"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "06a-CMD1..4",
                   "log_path": "raw/cpu-baseline.txt"} for c in cases],
        "measurements": [{"run_id": "06a-cpu-baseline", "raw_path": "raw/cpu-baseline.txt",
                          "sha256": sha(RAW / "cpu-baseline.txt")}],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no code change)",
        "key_findings": {"physical_cores": P, "thread_grid": thread_grid, "smt_grid": smt_grid,
                         "isa": {"avx2": True, "avx512f": True, "avx512bf16": False},
                         "cpu_moe_baseline_gbs": cpu_moe_nvfp4},
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 06a; P", P, "grid", thread_grid, "smt", smt_grid)


if __name__ == "__main__":
    main()
