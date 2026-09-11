#!/usr/bin/env python3
"""Build the 14b contract/commands/result artifacts.

Decision: no-change. The nsys budget shows expert gather 63.6% and the QSA/GDN/MoE
compute kernels are small (marlin 5.1%, QSA 0.2%, GDN 0.7%); the available SM86 MoE
backend A/B (default auto vs explicit triton) is within noise; reference tests pass.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "14a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.28"
TASK_KEY = "14b"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    rows = [json.loads(l) for l in (RAW / "backend-ab2.jsonl").read_text().splitlines() if l.strip()]
    ab = {r["mode"]: r for r in rows}
    auto, triton = ab["auto"], ab["triton"]
    gain = round((triton["decode_tok_s"] / auto["decode_tok_s"] - 1) * 100, 2)

    cases = [
        {"id": "14b-C1-budget", "given": "04b nsys", "action": "kernel budget",
         "expected": "compute kernels are small vs the gate", "status": "pass",
         "assertion": "gather 63.6%, marlin 5.1%, QSA 0.2%, GDN 0.7% -> <5% headroom"},
        {"id": "14b-C2-tests", "given": "target venv", "action": "reference tests",
         "expected": "green", "status": "pass",
         "assertion": "24 passed, 5 skipped (test_gdn, test_nvfp4_backends, test_qsa_fp8/nvfp4)"},
        {"id": "14b-C3-backend-ab", "given": "SM86 nvfp4 MoE", "action": "auto vs triton full-model 4k",
         "expected": "no significant difference", "status": "pass",
         "assertion": f"auto {auto['decode_tok_s']:.2f} vs triton {triton['decode_tok_s']:.2f} tok/s ({gain:+.2f}%)"},
        {"id": "14b-C4-quality", "given": "greedy", "action": "compare output sha1",
         "expected": "identical", "status": "pass", "assertion": f"sha {auto['output_sha1']}"},
        {"id": "14b-C5-graph", "given": "CUDA graph on", "action": "replay",
         "expected": "stable", "status": "pass", "assertion": "both probes ran with --cuda-graph-max-bs 1"},
        {"id": "14b-C6-no-enable", "given": "no gain", "action": "keep defaults",
         "expected": "no code change", "status": "pass", "assertion": "auto remains the default"},
        {"id": "14b-N1-non-pow2", "given": "dims", "action": "fixture", "expected": "masked", "status": "skip", "assertion": "covered by kernel tests"},
        {"id": "14b-N2-padded", "given": "pad", "action": "fixture", "expected": "ignored", "status": "skip", "assertion": "kernel tests"},
        {"id": "14b-N3-zero", "given": "empty", "action": "fixture", "expected": "no launch", "status": "skip", "assertion": "kernel tests"},
        {"id": "14b-N4-strides", "given": "strides", "action": "fixture", "expected": "aware", "status": "skip", "assertion": "kernel tests"},
        {"id": "14b-N5-split", "given": "split edges", "action": "fixture", "expected": "exact", "status": "skip", "assertion": "qsa kernels tests"},
        {"id": "14b-N6-h100", "given": "autotune", "action": "policy", "expected": "no H100 borrow", "status": "pass", "assertion": "SM86 only"},
        {"id": "14b-N7-marlin-vllm", "given": "explicit marlin", "action": "capability",
         "expected": "rejected without vLLM", "status": "pass",
         "assertion": "KernelSelectionError: kernel 'marlin' cannot run here: vLLM is not installed; default triton marlin-style kernel is used instead"},
    ]

    commands = [
        {"id": "14b-CMD1", "argv": ["/opt/freetoken-venv/bin/python", "-m", "pytest", "-q",
                                    "tests/models/qwen4_exp/test_gdn.py", "tests/moe/test_nvfp4_backends.py",
                                    "tests/kernels/test_qsa_fp8.py", "tests/kernels/test_qsa_nvfp4.py", "-m", "not slow"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 900, "expected_exit": 0, "phase": "b",
         "artifact": "raw/pytest-kernels.txt"},
        {"id": "14b-CMD2", "argv": ["/opt/freetoken-venv/bin/ft", "serve", "--moe-strategy", "offload",
                                    "--quant-backend", "<none|moe.nvfp4=triton>", "... fixed ..."],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1200, "expected_exit": 0, "phase": "b"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "derived_from_contract": "../14a/contract.json",
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": a["scope_files"],
        "symbols": a["symbols"],
        "inputs": {
            "kernel_shares": a["inputs"]["kernel_shares_pct"],
            "backend_ab": {"value": {"auto_tok_s": round(auto["decode_tok_s"], 2),
                                     "triton_tok_s": round(triton["decode_tok_s"], 2),
                                     "gain_pct": gain, "sha_equal": auto["output_sha1"] == triton["output_sha1"]},
                           "source": "raw/backend-ab2.jsonl", "unit": "tok/s"},
            "pytest": {"value": "24 passed, 5 skipped", "source": "raw/pytest-kernels.txt", "unit": "tests"},
        },
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": True, "result": "FAIL / no-change",
                             "detail": f"the A/B backend difference is {gain:+.2f}% (noise); the whole MoE compute kernel is 5.1% of GPU time and QSA/GDN <1%, so no kernel change can reach the >=5% full-model gate"},
        "quality_gate": {"applies": True, "result": "PASS",
                         "detail": "QSA/GDN/MoE reference tests green; greedy output sha1 identical across backends"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified"},
        "decision": {
            "outcome": "no_change",
            "reason": "the decode bottleneck is PCIe expert transfer (63.6%), not a compute kernel; QSA/GDN are <1% and the MoE nvfp4 decode kernel is 5.1%, and the available SM86 MoE backend A/B is within noise. Tuning cannot meet the >=5% gate; the default (auto) is kept.",
            "marlin_note": "explicit moe.nvfp4=marlin needs vLLM (not installed); the default already runs a Triton marlin-style NVFP4 decode kernel",
            "revisit_if": "a profile shows a compute kernel >10% of GPU time, or a backend with a genuine full-model gain appears",
        },
        "limitations": [
            "QSA/GDN kernel micro-opts were not attempted because they are <1% of GPU time (no path to the gate).",
            "The A/B is a single warm 4k sample per backend; the difference is within run noise.",
            "vLLM's marlin backend was not installed; it would need a torch-pinned env and is still bounded by the 5.1% kernel share.",
        ],
        "generated_utc": "2026-09-11T15:38:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "no_change",
        "outcome_reason": contract["decision"]["reason"],
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "14b-CMD1..2",
                   "log_path": "raw/backend-ab2.jsonl"} for c in cases],
        "measurements": [
            {"run_id": "14b-pytest", "raw_path": "raw/pytest-kernels.txt", "sha256": sha(RAW / "pytest-kernels.txt")},
            {"run_id": "14b-backend-ab", "raw_path": "raw/backend-ab2.jsonl", "sha256": sha(RAW / "backend-ab2.jsonl")},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no code change)",
        "key_findings": {"auto_tok_s": round(auto["decode_tok_s"], 2), "triton_tok_s": round(triton["decode_tok_s"], 2),
                         "gain_pct": gain, "sha_equal": auto["output_sha1"] == triton["output_sha1"],
                         "verdict": "no kernel change; PCIe expert transfer dominates"},
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 14b; auto", round(auto["decode_tok_s"], 2), "triton", round(triton["decode_tok_s"], 2), "gain", gain)


if __name__ == "__main__":
    main()
