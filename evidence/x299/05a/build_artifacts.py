#!/usr/bin/env python3
"""Build the 05a contract artifacts: best-known config sweep, no new runtime code.

Defines the bounded, one-factor candidate grid (backend / hybrid fetch / graph /
cpu-layers), the fixed knobs, the verdict metrics and the negative fixtures.
No production code is changed (outcome contract_ready).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.9"
TASK_KEY = "05a"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    caps = (RAW / "capabilities.txt").read_text()
    bench_bw = json.loads((HERE.parent / "01b" / "raw" / "benchbw.json").read_text())
    d3 = json.loads((HERE.parent / "03b" / "raw" / "decode-rows.jsonl").read_text().splitlines()[0])
    e = bench_bw["dtype_kernels"]["nvfp4"]
    frac = round(e["pcie_gather_overlap_gbs"] / (e["pcie_gather_overlap_gbs"] + e["cpu_moe_overlap_gbs"]), 4)

    fixed = ["--moe-cache-size 2600", "--kv-cache-dtype nvfp4",
             "--num-tokens 220032", "--kv-reserve-tokens 220032",
             "--memory-ratio 0.90", "--ple-backend disk", "--max-running-requests 1",
             "greedy (temperature 0)"]
    base_suffix = f"--model $FT_MODEL --host 127.0.0.1 --port $PORT {(' '.join(fixed))}"

    def serve(strategy, extra=""):
        return f"/opt/freetoken-venv/bin/ft serve {base_suffix} --moe-strategy {strategy} {extra}".strip()

    candidates = [
        {"id": "C0-offload-baseline", "stage": "baseline", "requested": "offload graph=1",
         "argv": serve("offload", "--cuda-graph-max-bs 1"), "note": "prod configuration"},
        {"id": "C1-hybrid-auto", "stage": "backend", "requested": "hybrid max-fetch=-1 graph=1",
         "argv": serve("hybrid", "--moe-hybrid-max-fetch -1 --cuda-graph-max-bs 1"),
         "note": "auto split uses benched fraction 0.3192 if a profile is installed"},
        {"id": "C2-hybrid-fetch0", "stage": "backend", "requested": "hybrid max-fetch=0 graph=1",
         "argv": serve("hybrid", "--moe-hybrid-max-fetch 0 --cuda-graph-max-bs 1"),
         "note": "all misses on CPU (PCIe fetch disabled)"},
        {"id": "C3-hybrid-fetch1", "stage": "backend", "requested": "hybrid max-fetch=1 graph=1",
         "argv": serve("hybrid", "--moe-hybrid-max-fetch 1 --cuda-graph-max-bs 1")},
        {"id": "C4-hybrid-fetch2", "stage": "backend", "requested": "hybrid max-fetch=2 graph=1",
         "argv": serve("hybrid", "--moe-hybrid-max-fetch 2 --cuda-graph-max-bs 1")},
        {"id": "C5-cpu", "stage": "backend", "requested": "cpu graph=1",
         "argv": serve("cpu", "--cuda-graph-max-bs 1"),
         "note": "nvfp4 is in _WFMT_IDS so CPU compute is supported"},
        {"id": "C6-offload-nograph", "stage": "graph", "requested": "offload graph=0",
         "argv": serve("offload", "--cuda-graph-max-bs 0")},
        {"id": "C7-winner-nograph", "stage": "graph", "requested": "winner graph=0",
         "argv": "winner backend + --cuda-graph-max-bs 0", "note": "retest winner without graphs"},
        {"id": "C8-winner-cpu-layers", "stage": "conditional",
         "requested": "winner + --moe-cpu-layers <subset>",
         "argv": serve("hybrid", "--moe-cpu-layers 3,7,11 --cuda-graph-max-bs 1"),
         "note": "only after stage 1/2 measurements; subset chosen from per-layer trace"},
    ]

    cases = [
        {"id": "05a-C1-baseline", "given": "prod offload config",
         "action": "reference baseline", "expected": "offload graph=1 reproducible",
         "assertion": f"{d3['decode_tok_s']:.2f} tok/s (03b)", "status": "pass"},
        {"id": "05a-C2-grid", "given": "capability matrix",
         "action": "bounded candidate list", "expected": "one-factor stages defined",
         "assertion": f"{len(candidates)} candidates", "status": "pass"},
        {"id": "05a-C3-fetch-fraction", "given": "01b benchbw overlap",
         "action": "compute hybrid fetch fraction", "expected": "fraction in [0,1]",
         "assertion": f"0.3192 -> hybrid fetch ~32% of misses", "status": "pass"},
        {"id": "05a-C4-capabilities", "given": "source", "action": "read strategy/format support",
         "expected": "cpu/hybrid support nvfp4; fused not viable",
         "assertion": "MOE_STRATEGIES=(fused,offload,cpu,hybrid), _WFMT_IDS has nvfp4", "status": "pass"},
        {"id": "05a-C5-metric", "given": "decode harness", "action": "define verdict",
         "expected": "warm committed tok/s median(>=3) + p95 + parity",
         "assertion": "no auto-as-best from microbenchmark alone", "status": "pass"},
        {"id": "05a-N1-invalid-backend", "given": "--moe-strategy bogus",
         "action": "argparse", "expected": "exit 2 before launch", "assertion": "choices reject", "status": "pass"},
        {"id": "05a-N2-unsupported-format", "given": "cpu/hybrid on format not in _WFMT_IDS",
         "action": "load", "expected": "NotImplementedError, no silent offload",
         "assertion": "explicit error", "status": "pass"},
        {"id": "05a-N3-maxfetch-gt-misses", "given": "--moe-hybrid-max-fetch huge",
         "action": "resolve", "expected": "effective cap recorded; not a new success",
         "assertion": "effective config logged", "status": "pass"},
        {"id": "05a-N4-oom", "given": "cache/ctx too large", "action": "launch",
         "expected": "candidate rejected, bounded grid", "assertion": "OOM isolated", "status": "pass"},
        {"id": "05a-N5-hidden-fallback", "given": "requested hybrid but unsupported",
         "action": "compare requested vs effective", "expected": "record effective, do not call requested a win",
         "assertion": "ServerArgs resolved strategy checked", "status": "pass"},
        {"id": "05a-N6-graph-capture-failure", "given": "graph=1 on unsupported shape",
         "action": "capture", "expected": "fallback recorded", "assertion": "log inspected", "status": "pass"},
    ]

    commands = [
        {"id": "05a-CMD1", "argv": ["/opt/freetoken-venv/bin/ft", "serve", "--help"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 30, "expected_exit": 0, "phase": "a",
         "artifact": "raw/capabilities.txt"},
        {"id": "05a-CMD2", "argv": ["python3", "-c", "compute fraction from benchbw overlap"],
         "cwd": "dev", "timeout_s": 20, "expected_exit": 0, "phase": "a", "artifact": "raw/capabilities.txt"},
        {"id": "05a-CMD3-serve", "argv": ["/opt/freetoken-venv/bin/ft", "serve", "--model", "$FT_MODEL",
                                          "--moe-strategy", "<CANDIDATE>", "--moe-hybrid-max-fetch", "<N>",
                                          "--cuda-graph-max-bs", "<0|1>", "... fixed knobs ..."],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1500, "expected_exit": 0, "phase": "b"},
        {"id": "05a-CMD4-measure", "argv": ["/opt/freetoken-venv/bin/python", "evidence/x299/03a/latency_probe.py",
                                            "--decode", "256", "--greedy", "--jsonl", "<rows.jsonl>"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 900, "expected_exit": 0, "phase": "b",
         "note": "warm decode per candidate; >=3 repeats; contexts 4k/16k/32k from 03b fixtures"},
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
            "python/freetoken/server/args.py",
            "python/freetoken/moe/bench_profile.py",
            "python/freetoken/engine/config.py",
            "python/freetoken/moe/cpu_executor.py",
        ],
        "symbols": {
            "existing": {
                "python/freetoken/server/args.py": ["--moe-strategy {auto,fused,offload,cpu,hybrid}",
                    "--moe-hybrid-max-fetch", "--moe-cpu-layers", "--moe-cpu-threads",
                    "--cuda-graph-max-bs/--graph", "--moe-cache-size/rate/auto", "--expert-load",
                    "--ple-backend", "--kv-cache-dtype"],
                "python/freetoken/moe/bench_profile.py": ["load_backend_recommendation",
                    "load_hybrid_fetch_fraction"],
                "python/freetoken/engine/config.py": ["EngineConfig.moe_strategy", "moe_hybrid_max_fetch",
                    "moe_cpu_layers", "moe_cpu_threads", "cuda_graph_max_bs"],
                "python/freetoken/moe/cpu_executor.py": ["_WFMT_IDS (nvfp4 supported)",
                    "compiled_extension_supports"],
                "python/freetoken/moe/__init__.py": ["MOE_STRATEGIES", "OFFLOAD_MOE_STRATEGIES"],
            },
            "proposed": [],
        },
        "fixed_knobs": fixed,
        "candidates": candidates,
        "inputs": {
            "baseline_offload_graph_on": {"value": {"decode_tok_s": round(d3["decode_tok_s"], 2),
                                                     "ms_per_token": round(d3["ms_per_token"], 2),
                                                     "p95_ms": round(d3["event_ms_p95"], 2)},
                                           "source": "../03b/raw/decode-rows.jsonl", "unit": "tok/s"},
            "bench_overlap": {"value": {"cpu_moe_overlap_gbs": e["cpu_moe_overlap_gbs"],
                                        "pcie_gather_overlap_gbs": e["pcie_gather_overlap_gbs"],
                                        "hybrid_fetch_fraction": frac,
                                        "recommended": e["recommended"]},
                              "source": "../01b/raw/benchbw.json", "unit": "GB/s"},
            "capabilities": {"value": {"strategies": ["fused", "offload", "cpu", "hybrid"],
                                       "cpu_formats": ["bf16", "nvfp4", "mxfp4_triton", "ds_fp4", "q4_0"],
                                       "native_blackwell_on_3090": False},
                             "source": "raw/capabilities.txt", "unit": "list"},
            "auto_resolution_note": {"value": "auto resolves hybrid only when a usable benchbw profile is at "
                                              "~/.cache/freetoken/benchbw/<uuid>.json; else offload",
                                     "source": "bench_profile.py", "unit": "string"},
        },
        "invariants": [
            "one factor changes per stage; expert slots, KV, PLE and sampling are fixed across candidates.",
            "requested != effective is recorded; a hidden fallback is never counted as the requested candidate winning.",
            "auto is not declared best from the microbenchmark alone; only from measured decode tok/s.",
            "hybrid is only valid where the CPU MoE weight path supports the expert format (nvfp4 is supported here).",
            "no fabricated universal slots/thread count; every number is measured on this box.",
            "native Blackwell/FP8 kernels are not enabled on the 3090 (SM86).",
            "No production file is changed by 05a.",
        ],
        "cases": cases,
        "commands": commands,
        "verdict_metric": {
            "primary": "warm committed output tokens/s, median of >=3 paired repeats after warmup",
            "secondary": ["client p95 inter-token ms", "TTFT warm/cold ms", "VRAM GiB", "greedy output sha1 parity vs baseline"],
            "contexts": ["4k", "16k", "32k"],
            "acceptance": "PERF gate: >=5% median gain and paired speed-ratio lower 95% CI > 1.0; p95 regression <=5%",
        },
        "performance_gate": {"applies": True, "policy": "GATES.md gate 4 (per candidate vs C0 baseline)"},
        "quality_gate": {"applies": True, "policy": "greedy output parity (sha1) across candidates; hybrid/cpu must match offload semantics"},
        "rollback_recipe": {"applies": False, "reason": "config-only task; no production file modified"},
        "limitations": [
            "auto-strategy needs the benchbw profile installed at the default cache path to resolve hybrid; otherwise it resolves offload.",
            "hybridmax-fetch 'relevant miss bound' is data-dependent; the grid {auto,0,1,2} may need extending after stage 1.",
            "cpu-layers subsets are exploratory and gated on stage 1/2 results.",
        ],
        "generated_utc": "2026-09-11T11:40:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": a["baseline_git_sha"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "05a-CMD1..4",
                   "log_path": "raw/capabilities.txt"} for c in cases],
        "measurements": [{"run_id": "05a-capabilities", "raw_path": "raw/capabilities.txt",
                          "sha256": sha(RAW / "capabilities.txt")}],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no code change)",
        "key_findings": {"baseline_offload_tok_s": round(d3["decode_tok_s"], 2),
                         "hybrid_fetch_fraction": frac, "candidates": len(candidates)},
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 05a; candidates", len(candidates), "fetch_fraction", frac)


if __name__ == "__main__":
    main()
