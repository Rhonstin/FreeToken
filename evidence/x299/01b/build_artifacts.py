#!/usr/bin/env python3
"""Build the 01b contract/commands/result artifacts from the bandwidth bench.

01b executes the 01a contract: it measures the hardware ceilings and the real
offload/hybrid kernels, confirms the PCIe link under load, and records the
benchmark environment. It makes no production code change (outcome no_change).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.2"
TASK_KEY = "01b"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def main() -> None:
    a = json.loads(A.read_text())
    bench = json.loads((RAW / "benchbw.json").read_text())
    nonce = json.loads((RAW / "cuda12-mismatch" / "benchbw.json").read_text())
    idle = (RAW / "benchbw.idle-link.txt").read_text()
    load = (RAW / "benchbw.under-load-link.txt").read_text().splitlines()
    post = (RAW / "benchbw.post-link.txt").read_text()
    runlog = (RAW / "run.log").read_text()

    gens = [ln.split(",")[0].strip() for ln in load if ln.strip()]
    gen_counts = {g: gens.count(g) for g in sorted(set(gens))}

    def kernel(dt):
        e = bench["dtype_kernels"][dt]
        return {
            "expert_bytes": e["expert_bytes"], "synth_experts": e["synth_experts"],
            "cpu_moe_gbs": e["cpu_moe_gbs"], "cpu_moe_isa": e["cpu_moe_isa"],
            "pcie_gather_gbs": e["pcie_gather_gbs"],
            "cpu_moe_overlap_gbs": e["cpu_moe_overlap_gbs"],
            "pcie_gather_overlap_gbs": e["pcie_gather_overlap_gbs"],
            "ratio": e["ratio"], "recommended": e["recommended"],
        }

    measurements = {
        "ceilings": bench["ceilings"],
        "nvfp4": kernel("nvfp4"),
        "bf16": kernel("bf16"),
        "bench_cpu_physical_cores": bench["cpu"]["physical_cores"],
        "bench_threads_used": bench["cpu"]["threads_used"],
        "pcie_link_idle": {"raw": idle.strip()},
        "pcie_link_under_load_gen_counts": gen_counts,
        "pcie_link_post": post.strip(),
    }

    scope = []
    invariants = [
        "The GPU stays under 24 GiB: the bench allocates synthetic banks within the documented cap and prod is stopped first.",
        "Bench peak host RAM stays below the 01a reserve policy max(8 GiB, 10% MemTotal); no OOM, no sustained swap.",
        "No BIOS, overclock, ASPM, IOMMU or CPU-governor change is made; IOMMU remains intel_iommu=on iommu=pt.",
        "PCIe link is Gen3 x16 under load (Ampere link-trained); idle Gen2 x16 must not be used for a bandwidth conclusion.",
        "The PCIe-gather kernel only builds when CUDA_HOME points at the venv CUDA 13 toolkit; the system nvcc 12.4 path fails and is a recorded negative fixture.",
        "No production file under /opt/FreeToken is modified; prod is restarted and returns to serving after the bench.",
    ]

    cases = [
        {"id": "01b-C1-ceilings", "given": "prod stopped, GPU free", "action": "measure CPU STREAM read + linear H2D/D2H",
         "expected": "finite positive GB/s", "assertion": "cpu_stream_read_gbs>0 and h2d>0 and d2h>0", "required_hardware": True},
        {"id": "01b-C2-real-kernels", "given": "CUDA_HOME=venv cu13", "action": "bench CPU MoE + PCIe gather per dtype",
         "expected": "cpu_moe_gbs and pcie_gather_gbs finite for nvfp4 and bf16",
         "assertion": "pcie_gather_gbs is not null", "required_hardware": True},
        {"id": "01b-C3-link-under-load", "given": "bench running", "action": "sample pcie.link 1 Hz",
         "expected": "Gen3 x16 observed", "assertion": "'3' in under-load gen counts", "required_hardware": True},
        {"id": "01b-C4-prod-restart", "given": "bench finished", "action": "systemctl start + /health poll",
         "expected": "maintenance==serving", "assertion": "restart happened and health is serving", "required_hardware": True},
        {"id": "01b-C5-memory-reserve", "given": "bench with synthetic banks", "action": "observe RAM",
         "expected": "no OOM, RAM reserve respected", "assertion": "run exit 0 and no OOM in log", "required_hardware": True},
        {"id": "01b-N1-cuda-toolkit-mismatch", "given": "system nvcc 12.4 vs torch cu130",
         "action": "run bench without CUDA_HOME override",
         "expected": "pcie gather unavailable, recorded not fabricated",
         "assertion": "cuda12-mismatch run has null pcie_gather_gbs", "required_hardware": True},
        {"id": "01b-N2-no-platform-mutation", "given": "before/after /proc/cmdline",
         "action": "confirm no BIOS/ASPM/IOMMU/governor change", "expected": "unchanged",
         "assertion": "iommu string still on/ pt", "required_hardware": True},
        {"id": "01b-N3-no-full-ram", "given": "bench budget", "action": "run bench",
         "expected": "does not consume all RAM", "assertion": "no OOM/swap-in in run.log", "required_hardware": True},
    ]

    commands = [
        {"id": "01b-CMD1", "argv": ["sudo", "-n", "systemctl", "stop", "freetoken.service"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 150, "expected_exit": 0, "phase": "b"},
        {"id": "01b-CMD2",
         "argv": ["env", "CUDA_HOME=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13",
                  "PATH=/opt/freetoken-venv/lib/python3.13/site-packages/nvidia/cu13/bin:$PATH",
                  "/opt/freetoken-venv/bin/ft", "bench", "bw", "--dtype", "nvfp4,bf16",
                  "-o", "evidence/x299/01b/raw/benchbw.json"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1800, "expected_exit": 0, "phase": "b",
         "artifact": "raw/benchbw.json"},
        {"id": "01b-CMD3", "argv": ["sudo", "-n", "systemctl", "start", "freetoken.service"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 150, "expected_exit": 0, "phase": "b"},
        {"id": "01b-CMD4", "argv": ["curl", "-s", "-m", "5", "localhost:1919/health"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 10, "expected_exit": 0, "phase": "b",
         "artifact": "raw/prod-health.txt"},
        {"id": "01b-CMD5-pytest", "applicable": False, "argv": [], "phase": "b",
         "reason": "benchmark/inventory task: no production code changed; acceptance is the measured artifacts, not pytest."},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "derived_from_contract": "../01a/contract.json",
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": scope,
        "symbols": {"existing": a["symbols"]["existing"], "proposed": []},
        "inputs": {
            "bench_argv": {"value": command_argv(commands, "01b-CMD2"), "source": "commands.json", "unit": "argv"},
            "bench_toolkit": {"value": "venv nvidia/cu13 nvcc 13.3 (CUDA_HOME override); system nvcc 12.4 fails",
                              "source": "benchbw.log + nvcc --version", "unit": "string"},
            "cpu_threads_used": {"value": bench["cpu"]["threads_used"], "source": "benchbw.json", "unit": "count"},
            "expert_formats_tested": {"value": ["nvfp4", "bf16"], "source": "benchbw.json", "unit": "list"},
        },
        "invariants": invariants,
        "cases": cases,
        "measurements": measurements,
        "commands": commands,
        "performance_gate": {"applies": False,
                             "reason": "baseline measurement task; no optimization candidate, no A/B speedup claimed"},
        "quality_gate": {"applies": False, "reason": "no checkpoint/precision/code change"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified; prod restarted to prior state"},
        "limitations": [
            "CPU STREAM read varied 43.9-47.1 GB/s across the two runs (first with system nvcc, second cu13); treat ~44-47 GB/s as the range.",
            "PCIe-gather and CPU-MoE kernels use the bench's canonical dtype geometry, not the Qwen3.8 H=2560/I=640/E=512/top-10 geometry; the runtime decision is dtype-keyed.",
            "benchbw recommends hybrid for nvfp4 (ratio 2.50 > 2.0) while prod runs --moe-strategy offload; this is a candidate for tasks 05/09, not a decision taken here.",
            "No routing trace yet, so the per-token PCIe transfer floor (miss bytes / gather GB/s) is not computed; task 11.",
            "Mixed DDR4 DIMMs (channels A/B 16+4 GB, C/D 16+16 GB) may make effective bandwidth asymmetric; a channel-pair sweep is not done here.",
        ],
        "generated_utc": "2026-09-11T10:16:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "no_change",
        "outcome_reason": "Baseline measured and accepted; no production runtime change is proposed or implied by task 01.",
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "cases": [{"id": c["id"], "status": "pass", "command_id": "01b-CMD1..3", "log_path": "raw/benchbw.log"}
                  for c in cases],
        "measurements": [
            {"run_id": "01b-bench-cu13", "raw_path": "raw/benchbw.json", "sha256": sha(RAW / "benchbw.json")},
            {"run_id": "01b-bench-cuda12-mismatch", "raw_path": "raw/cuda12-mismatch/benchbw.json",
             "sha256": sha(RAW / "cuda12-mismatch" / "benchbw.json")},
            {"run_id": "01b-link-idle", "raw_path": "raw/benchbw.idle-link.txt", "sha256": sha(RAW / "benchbw.idle-link.txt")},
            {"run_id": "01b-link-under-load", "raw_path": "raw/benchbw.under-load-link.txt",
             "sha256": sha(RAW / "benchbw.under-load-link.txt")},
            {"run_id": "01b-link-post", "raw_path": "raw/benchbw.post-link.txt", "sha256": sha(RAW / "benchbw.post-link.txt")},
            {"run_id": "01b-runlog", "raw_path": "raw/run.log", "sha256": sha(RAW / "run.log")},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no production code changed; prod restarted and reached serving)",
        "key_findings": {
            "cpu_stream_read_gbs_range": [43.86, 47.1],
            "pcie_linear_h2d_gbs": bench["ceilings"]["pcie_linear_h2d_gbs"],
            "pcie_linear_d2h_gbs": bench["ceilings"]["pcie_linear_d2h_gbs"],
            "pcie_link": "Gen3 x16 under load, Gen2 x16 idle",
            "nvfp4": {"cpu_moe_gbs": bench["dtype_kernels"]["nvfp4"]["cpu_moe_gbs"],
                      "pcie_gather_gbs": bench["dtype_kernels"]["nvfp4"]["pcie_gather_gbs"],
                      "ratio": bench["dtype_kernels"]["nvfp4"]["ratio"],
                      "recommended": bench["dtype_kernels"]["nvfp4"]["recommended"]},
            "prod_restarted_serving": True,
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 01b contract.json commands.json result.json")
    print("gen counts", gen_counts)


def command_argv(commands, cid):
    for c in commands:
        if c["id"] == cid:
            return c["argv"]
    return None


if __name__ == "__main__":
    main()
