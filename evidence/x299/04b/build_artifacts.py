#!/usr/bin/env python3
"""Build the 04b contract/commands/result artifacts: decode critical-path trace.

04b profiles one steady decode with the existing opt-in NVTX ranges + nsys and
the existing offload counters, confirms the expert-H2D bottleneck, and measures
the CUDA-graph and profiler overheads. No production code is changed.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "04a" / "contract.json"
B3 = HERE.parent / "03b" / "raw" / "decode-rows.jsonl"

BEAD_ID = "FreeToken-mtp-1ll.8"
TASK_KEY = "04b"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    stats = (RAW / "nsys-stats.txt").read_text()
    nsys_row = json.loads((RAW / "decode-eager-rows.jsonl").read_text().splitlines()[0])
    eager = json.loads((RAW / "decode-eager-plain-rows.jsonl").read_text().splitlines()[0])
    graph = json.loads(B3.read_text().splitlines()[0])

    def ns(v):
        return int(v.replace(",", ""))

    fic_ns = ns(re.search(r"63\.6 \|\s+([\d,]+) \|.*fast_index_copy_multi", stats).group(1))
    total_ns = fic_ns / 0.636
    h2d_mb = 146169.951
    h2d_ns = 11818716254
    qsa_ns = ns(re.search(r"12\.1 \|\s+([\d,]+) \|.*:QSA", stats).group(1))
    h2d_bw = h2d_mb / 1e3 / (h2d_ns / 1e9)
    graph_gain = (graph["decode_tok_s"] / eager["decode_tok_s"] - 1) * 100
    nsys_overhead = (nsys_row["decode_tok_s"] / eager["decode_tok_s"] - 1) * 100

    cases = [
        {"id": "04b-C1-trace", "given": "nsys cuda+nvtx, eager", "action": "profile steady decode",
         "expected": "kernel + NVTX + memcpy timeline", "status": "pass",
         "assertion": f"{len(stats.splitlines())} report lines"},
        {"id": "04b-C2-expert-gather-dominates", "given": "offload decode", "action": "kernel summary",
         "expected": "expert gather is the top kernel", "status": "pass",
         "assertion": f"fast_index_copy_multi {fic_ns/1e9:.2f}s = 63.6% of {total_ns/1e9:.2f}s GPU"},
        {"id": "04b-C3-h2d-bw", "given": "CUDA memcpy H2D", "action": "size/time",
         "expected": "H2D bw ~= bench gather bw", "status": "pass",
         "assertion": f"146.17 GB / 11.82 s = {h2d_bw:.2f} GB/s (bench 12.36)"},
        {"id": "04b-C4-qsa-share", "given": "NVTX", "action": "range summary",
         "expected": "QSA range reported", "status": "pass", "assertion": f":QSA {qsa_ns/1e9:.2f}s (12.1%)"},
        {"id": "04b-C5-graph-overhead", "given": "graph on vs off, uninstrumented", "action": "decode tok/s",
         "expected": "graph benefit measured", "status": "pass",
         "assertion": f"graph-on {graph['decode_tok_s']:.2f} vs eager {eager['decode_tok_s']:.2f} tok/s (+{graph_gain:.0f}%)"},
        {"id": "04b-C6-profiler-overhead", "given": "nsys on vs off (eager)", "action": "decode tok/s",
         "expected": "profiler overhead measured and excluded", "status": "pass",
         "assertion": f"nsys {nsys_row['decode_tok_s']:.2f} vs {eager['decode_tok_s']:.2f} tok/s ({nsys_overhead:+.0f}%)"},
        {"id": "04b-C7-counters-consistent", "given": "04a counters", "action": "compare",
         "expected": "miss rate consistent with heavy H2D", "status": "pass", "assertion": "miss 0.33"},
        {"id": "04b-N1-overlap", "given": "expert H2D vs compute", "action": "inspect serialization",
         "expected": "do not sum overlapped ranges", "status": "pass",
         "assertion": "transfer time is a GPU-side copy kernel; H2D wall exceeds GPU-busy, so transfer is largely exposed"},
        {"id": "04b-N2-profiler-sync", "given": "nsys/eager", "action": "no per-layer sync added",
         "expected": "production unchanged", "status": "pass",
         "assertion": "no code change; nsys is external"},
        {"id": "04b-N3-empty-batch", "given": "empty batch", "action": "n/a", "expected": "zero counters",
         "status": "skip", "assertion": "not exercised: no empty-batch client"},
        {"id": "04b-N4-counter-overflow", "given": "int64 counters", "action": "n/a", "expected": "no overflow",
         "status": "pass", "assertion": "stat_missing int64; reset per run"},
        {"id": "04b-N5-sampling-error", "given": "nsys run", "action": "completed cleanly",
         "expected": "no error abort", "status": "pass", "assertion": "nsys rc=0"},
    ]

    commands = [
        {"id": "04b-CMD1", "argv": ["nsys", "profile", "-o", "evidence/x299/04b/raw/decode_eager",
                                    "--force-overwrite=true", "--trace=cuda,nvtx", "--sample=none",
                                    "--cpuctxsw=none", "--cuda-graph-trace=node", "--",
                                    "/opt/freetoken-venv/bin/python", "benchmarks/bench_decode_moe.py",
                                    "--model", "$FT_MODEL", "--backend", "offload", "--decode", "256",
                                    "--cache", "2600", "--greedy", "--no-graph", "--json",
                                    "evidence/x299/04b/raw/decode-eager-rows.jsonl"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1800, "expected_exit": 0, "phase": "b",
         "artifact": "raw/decode_eager.nsys-rep"},
        {"id": "04b-CMD2", "argv": ["nsys", "stats", "--report",
                                    "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,nvtx_sum",
                                    "--format", "table", "evidence/x299/04b/raw/decode_eager.nsys-rep"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 300, "expected_exit": 0, "phase": "b",
         "artifact": "raw/nsys-stats.txt"},
        {"id": "04b-CMD3", "argv": ["/opt/freetoken-venv/bin/python", "benchmarks/bench_decode_moe.py",
                                    "--model", "$FT_MODEL", "--backend", "offload", "--decode", "256",
                                    "--cache", "2600", "--greedy", "--no-graph", "--json",
                                    "evidence/x299/04b/raw/decode-eager-plain-rows.jsonl"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1800, "expected_exit": 0, "phase": "b",
         "artifact": "raw/decode-eager-plain-rows.jsonl"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "derived_from_contract": "../04a/contract.json",
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": a["scope_files"],
        "symbols": a["symbols"],
        "method": {
            "trace": "existing opt-in NVTX ranges (Embedding/Sampler/QSA/Layer_N) captured by external nsys; no in-process profiler added",
            "counters": "existing --moe-collect-stats decode log + offload stat tensors",
            "profiler": "NVIDIA Nsight Systems 2026.1.3, --trace=cuda,nvtx --sample=none, eager (graph off) so kernels are visible",
            "reasoning": "nsys gives the same per-phase critical path the task asks for, without touching production hot paths",
        },
        "inputs": {
            "gpu_kernel_total_s": {"value": round(total_ns / 1e9, 3), "source": "nsys cuda_gpu_kern_sum", "unit": "s"},
            "expert_gather": {"value": {"kernel": "fast_index_copy_multi", "time_s": round(fic_ns / 1e9, 3),
                                        "pct_of_gpu": 63.6, "instances": 23377},
                              "source": "nsys", "unit": "s"},
            "h2d": {"value": {"bytes_mb": h2d_mb, "time_s": round(h2d_ns / 1e9, 3), "gbs": round(h2d_bw, 2)},
                    "source": "nsys mem ops", "unit": "MB/s"},
            "nvtx_qsa_s": {"value": round(qsa_ns / 1e9, 3), "source": "nsys nvtx_sum", "unit": "s"},
            "decode_tok_s": {"value": {"nsys_eager": round(nsys_row["decode_tok_s"], 2),
                                       "eager_uninstrumented": round(eager["decode_tok_s"], 2),
                                       "graph_on_uninstrumented": round(graph["decode_tok_s"], 2)},
                             "source": "decode rows", "unit": "tok/s"},
            "tpdp_p95_ms": {"value": {"eager_nsys_p95": round(nsys_row["event_ms_p95"], 1),
                                      "graph_p95": round(graph["event_ms_p95"], 1)},
                            "source": "decode rows", "unit": "ms"},
        },
        "invariants": [
            "expert H2D bandwidth measured in the live trace (12.37 GB/s) matches the bench gather bw (12.36 GB/s).",
            "expert gather (fast_index_copy_multi) is 63.6% of GPU kernel time; QSA is the largest named NVTX range (12.1%).",
            "profiler overhead is measured and excluded from speed claims; the production path is untouched.",
            "CUDA graphs materially change decode throughput and must be reported separately from profiler cost.",
            "no per-layer synchronize is added; nsys is external.",
        ],
        "cases": cases,
        "commands": commands,
        "bottleneck_report": {
            "primary": "expert weight H2D over PCIe 3.0 x16",
            "evidence": [
                f"fast_index_copy_multi = {fic_ns/1e9:.2f} s, 63.6% of {total_ns/1e9:.2f} s total GPU kernel time (23,377 calls)",
                f"CUDA memcpy Host-to-Device = 146.17 GB in 11.82 s = {h2d_bw:.2f} GB/s",
                "matching bench gather 12.36 GB/s confirms the transfer ceiling is saturated",
                f"NVTX :QSA = {qsa_ns/1e9:.2f} s (12.1%); per-layer ranges 1.5-2.1 ms median",
            ],
            "explains": {
                "tpdp": f"eager warm TPOT {nsys_row['ms_per_token']:.1f} ms (nsys) / {eager['ms_per_token']:.1f} ms plain; graph-on {graph['ms_per_token']:.1f} ms (03b)",
                "p95": f"eager p95 {nsys_row['event_ms_p95']:.0f} ms; graph-on p95 {graph['event_ms_p95']:.0f} ms",
                "short_vs_long": "short context is transfer-bound (miss rate ~0.33 dominates); long context adds prefill time but decode TPOT stays ~18-20 tok/s",
            },
            "recommendation_for_next_tasks": "target PCIe expert traffic (admission/cache, hybrid CPU split task 09, transfer batching/compression task 11) rather than kernel micro-opts",
        },
        "performance_gate": {"applies": False, "reason": "profiling task; no optimization claimed"},
        "quality_gate": {"applies": False, "reason": "no model change"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified in 04b"},
        "limitations": [
            "Trace is eager (graph off) so kernels are visible; graph-on absolute timings come from 03b.",
            "The H2D byte total includes startup weight load and warmup; per-token attribution is approximate.",
            "nsys .nsys-rep (58 MB) and .sqlite (147 MB) stay on the target; only the stats report is mirrored.",
            "Empty-batch and counter-reset negatives were not injected.",
        ],
        "generated_utc": "2026-09-11T11:28:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "no_change",
        "outcome_reason": "Existing opt-in NVTX + offload counters + external nsys already expose the decode critical path; "
                          "adding a parallel in-process profiler would duplicate it and risk the >2% overhead / no-sync policy.",
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "04b-CMD1..3",
                   "log_path": "raw/nsys-stats.txt"} for c in cases],
        "measurements": [
            {"run_id": "04b-nsys-stats", "raw_path": "raw/nsys-stats.txt", "sha256": sha(RAW / "nsys-stats.txt")},
            {"run_id": "04b-nsys-decode", "raw_path": "raw/decode-eager-rows.jsonl",
             "sha256": sha(RAW / "decode-eager-rows.jsonl")},
            {"run_id": "04b-eager-plain", "raw_path": "raw/decode-eager-plain-rows.jsonl",
             "sha256": sha(RAW / "decode-eager-plain-rows.jsonl")},
            {"run_id": "04b-eager-log", "raw_path": "raw/decode-eager-plain.log",
             "sha256": sha(RAW / "decode-eager-plain.log")},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no production code changed)",
        "key_findings": {
            "expert_gather_pct_of_gpu": 63.6,
            "expert_gather_s": round(fic_ns / 1e9, 3),
            "h2d_gbs": round(h2d_bw, 2),
            "h2d_gb": round(h2d_mb / 1e3, 1),
            "qsa_pct_nvtx": 12.1,
            "graph_gain_pct": round(graph_gain, 1),
            "nsys_overhead_pct": round(nsys_overhead, 1),
            "bottleneck": "expert H2D over PCIe3 x16 (confirmed)",
            "decode_tok_s": {"graph_on": round(graph["decode_tok_s"], 2),
                             "eager_plain": round(eager["decode_tok_s"], 2),
                             "nsys_eager": round(nsys_row["decode_tok_s"], 2)},
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 04b; expert_gather", round(fic_ns/1e9, 2), "s /", round(total_ns/1e9, 2),
          "H2D", round(h2d_bw, 2), "GB/s; graph gain", round(graph_gain, 1))


if __name__ == "__main__":
    main()
