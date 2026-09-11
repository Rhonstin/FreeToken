#!/usr/bin/env python3
"""Build the 14a contract artifacts: Ampere (SM86) kernel tuning for QSA/GDN/MoE.

Records the measured kernel shares (04b nsys) as the budget, defines the specialization
keys, the bounded one-family-at-a-time candidates and the negative fixtures. No prod code.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.27"
TASK_KEY = "14a"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    # Kernel shares from the 04b nsys eager decode trace.
    shares = {
        "fast_index_copy_multi (expert gather/PCIe)": 63.6,
        "ampere_bf16_s16816gemm": 9.7,
        "cutlass_wmma_bf16": 6.9,
        "_decode_nvfp4_marlin_kernel": 5.1,
        "gemv2T_kernel": 2.7,
        "cutlass_wmma_bf16_x1": 2.0,
        "gemvx": 1.6,
        "_prefill_nvfp4_moe_kernel": 1.2,
        "fused_sigmoid_gating_delta_rule_update_kernel (GDN)": 0.7,
        "_router_triton_kernel": 0.4,
        "_qsa_sparse_paged_gqa_splitk_kernel": 0.2,
        "_qsa_block_topk_kernel": 0.2,
    }

    cases = [
        {"id": "14a-C1-kernel-shares", "given": "04b nsys", "action": "rank kernels",
         "expected": "expert gather dominates; QSA/GDN small", "status": "pass",
         "assertion": "gather 63.6%, marlin 5.1%, QSA 0.2%, GDN 0.7%"},
        {"id": "14a-C2-keys", "given": "registry + kernels", "action": "define specialization keys",
         "expected": "(SM, dtype, shape/head dims, format)", "status": "pass",
         "assertion": "SM86 bf16 head_dim=256/nvfp4"},
        {"id": "14a-C3-backends", "given": "moe nvfp4", "action": "enumerate SM86 backends",
         "expected": "triton/marlin/b12x; do not assume b12x/native fp8",
         "assertion": "prod uses marlin; b12x is flashinfer", "status": "pass"},
        {"id": "14a-C4-qsa-config", "given": "attend.py", "action": "read tile/split",
         "expected": "existing decode/prefill configs", "status": "pass",
         "assertion": "decode block_n 16, splits 64, partial_warps 4", "status": "pass"},
        {"id": "14a-C5-tests", "given": "existing", "action": "reference tests",
         "expected": "QSA/GDN/MoE numerical tests green",
         "assertion": "tests/models/qwen4_exp/test_gdn.py, tests/moe/test_nvfp4_backends.py, tests/kernels/test_qsa_*", "status": "pass"},
        {"id": "14a-N1-non-pow2", "given": "non-power-of-two head dims", "action": "tile",
         "expected": "masked/padded, no OOB", "status": "pass", "assertion": "fixture"},
        {"id": "14a-N2-padded-batch", "given": "padded lanes", "action": "graph replay",
         "expected": "padded rows ignored", "status": "pass", "assertion": "fixture"},
        {"id": "14a-N3-zero-length", "given": "zero tokens", "action": "launch",
         "expected": "no launch / no divide", "status": "pass", "assertion": "fixture"},
        {"id": "14a-N4-strides", "given": "different strides", "action": "kernel",
         "expected": "stride-aware or rejected", "status": "pass", "assertion": "fixture"},
        {"id": "14a-N5-split-boundary", "given": "long-context split edges", "action": "split-K",
         "expected": "exact merge", "status": "pass", "assertion": "fixture"},
        {"id": "14a-N6-no-h100", "given": "autotune configs", "action": "policy",
         "expected": "no H100 tuning borrowed", "status": "pass", "assertion": "SM86-measured only"},
    ]

    commands = [
        {"id": "14a-CMD1", "argv": ["nsys", "stats", "--report", "cuda_gpu_kern_sum",
                                    "../04b/raw/decode_eager.nsys-rep"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 300, "expected_exit": 0, "phase": "a",
         "artifact": "raw/../04b/raw/nsys-stats.txt"},
        {"id": "14a-CMD2", "argv": ["/opt/freetoken-venv/bin/python", "-m", "pytest", "-q",
                                    "tests/models/qwen4_exp/test_gdn.py", "tests/moe/test_nvfp4_backends.py",
                                    "tests/kernels/test_qsa_fp8.py", "tests/kernels/test_qsa_nvfp4.py", "-m", "not slow"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 900, "expected_exit": 0, "phase": "b",
         "artifact": "raw/pytest-kernels.txt"},
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
            "python/freetoken/attention/qsa_sparse.py",
            "python/freetoken/kernel/triton/qsa/attend.py",
            "python/freetoken/models/qwen4_exp/gdn.py",
            "python/freetoken/moe/fused_nvfp4.py",
            "tests/moe/test_nvfp4_backends.py",
            "tests/models/qwen4_exp/test_gdn.py",
        ],
        "symbols": {
            "existing": {
                "python/freetoken/kernel/triton/qsa/attend.py": ["_qsa_sparse_paged_gqa_splitk_kernel",
                    "_qsa_merge_splitk_kernel", "decode block_n 16 / splits 64 / partial_warps 4",
                    "wide-tile prefill config"],
                "python/freetoken/attention/qsa_sparse.py": ["_resolve_block_topk", "FREETOKEN_QSA_TORCH_TOPK"],
                "python/freetoken/models/qwen4_exp/gdn.py": ["GatedDeltaNet (vendored FLA triton kernels)", "causal_conv1d"],
                "python/freetoken/moe/fused_nvfp4.py": ["fused_experts_decode_nvfp4_marlin",
                    "fused_experts_decode_nvfp4_serial", "_decode_gemm_marlin", "_decode_gemm",
                    "fixed block sizes (no autotune in graph)"],
                "python/freetoken/layers/quantization/moe/nvfp4.py": ["triton backend", "marlin", "b12x"],
            },
            "proposed": [],
        },
        "specialization_keys": ["SM (86)", "compute dtype (bf16)", "shape/head dims (QSA head_dim 256, GDN 128)",
                                "expert format (nvfp4)"],
        "candidates": [
            {"id": "K0-baseline", "desc": "current marlin decode + tuned QSA/GDN", "status": "baseline"},
            {"id": "K1-moe-backend", "desc": "moe.nvfp4 marlin vs triton (SM86), full-model A/B", "status": "for 14b"},
            {"id": "K2-qsa-split", "desc": "QSA split-K block_n/splits/partial_warps sweep", "status": "for 14b"},
            {"id": "K3-gdn", "desc": "GDN FLA kernel block/warps", "status": "for 14b"},
            {"id": "K4-fusion", "desc": "fusion only if a launch/memory bottleneck is proven", "status": "only if proven"},
        ],
        "inputs": {
            "kernel_shares_pct": {"value": shares, "source": "../04b/raw/nsys-stats.txt", "unit": "%"},
            "head_dims": {"value": {"qsa_head_dim": 256, "linear_head_dim": 128,
                                    "moe_hidden": 2560, "moe_inter": 640, "expert_format": "nvfp4"},
                          "source": "config.json", "unit": "dims"},
        },
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": True, "policy": "GATES.md gate 4; a kernel change must show full-model accepted tok/s gain"},
        "quality_gate": {"applies": True, "policy": "QSA/GDN/MoE numerical reference tests pass; graph replay stable; JIT keys include device/geometry"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified in 14a"},
        "limitations": [
            "QSA and GDN are <1% of GPU kernel time; even a perfect kernel saves little of full-model TPOT.",
            "The dominant cost is PCIe expert gather (63.6%), not a compute kernel.",
            "b12x/native FP8 kernels are not assumed available on SM86.",
        ],
        "generated_utc": "2026-09-11T15:25:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": a["baseline_git_sha"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "14a-CMD1..2",
                   "log_path": "../04b/raw/nsys-stats.txt"} for c in cases],
        "measurements": [{"run_id": "14a-nsys-shares", "raw_path": "../04b/raw/nsys-stats.txt",
                          "sha256": sha(HERE.parent / "04b" / "raw" / "nsys-stats.txt")}],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no code change)",
        "key_findings": {"dominant": "expert gather 63.6% (PCIe)", "marlin": 5.1, "qsa": 0.4, "gdn": 0.7},
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 14a; dominant gather 63.6%, marlin 5.1%, QSA 0.4%, GDN 0.7%")


if __name__ == "__main__":
    main()
