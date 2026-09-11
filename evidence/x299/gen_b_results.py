#!/usr/bin/env python3
"""Generate no_change result.json for the remaining P2 b-tasks (18,20,22,26,38)."""
from __future__ import annotations

import json
import pathlib

BASE = pathlib.Path("/opt/FreeToken-mtp/evidence/x299")
HW = "7ef87cf9868386b006cdf4384feca80a41e186dae7a7dbd050c0e5c0d907a289"

RESULTS = {
    "18": {
        "plan": "09b", "bead": "FreeToken-mtp-1ll.18",
        "title": "Calibrate the hybrid split by real expert time - execution and acceptance",
        "measurements": {
            "fetch_sweep_4k_05b": {"fetch0": 14.04, "fetch1": 24.32, "fetch2": 26.12, "note": "05b; fetch2 edge at 4k did not hold at 16k/32k"},
            "prod_shape_49": {"fetch1_16k": 25.20, "fetch1_64k": 25.09, "adopted_16k": 27.95},
            "offload_same_shape": {"16k": 21.88, "64k": 21.91}},
        "analysis": ["Static --moe-hybrid-max-fetch 1 is the best robust choice; the ideal-bandwidth ratio (0.3192) is already captured by fetch1.", "An adaptive/hysteresis policy was not implemented because no measurement showed a benefit beyond noise over static fetch1, and the plan forbids per-layer synchronous readback."],
        "decision": "no_change: keep the static hybrid fetch=1 (already adopted in 49); no adaptive max-fetch introduced.",
        "evidence": ["../../49/result.json", "raw/pytest-target.txt", "../../05b/result.json"],
        "limits": ["Adaptive policy not built; concluded from measured static sweep + prod-shape validation."]},
    "20": {
        "plan": "10b", "bead": "FreeToken-mtp-1ll.20",
        "title": "Improve cache admission from routing traces only - execution and acceptance",
        "measurements": {"target_test_offload": "30 passed", "pcie_h2d_gb_s": 12.2, "expert_gather_share_04b": 0.636,
                         "locality_12": {"repetitive": 0.994, "random_code": 0.0004}},
        "analysis": ["There is no per-layer expert-id trace collector in the tree; the plan requires an offline replay but the trace source does not exist.", "H2D is byte-saturated (04b): better admission helps only if it reduces EXPOSED bytes, and the measured CPU/PCIe overlap already hides part of it.", "Locality is workload-dependent (12): realistic prompts are near the random regime where LRU has nothing to exploit."],
        "decision": "no_change: keep the default LRU admission; no frequency/quota policy adopted without a demonstrable exposed-PCIe reduction.",
        "evidence": ["raw/pytest-target.txt", "../../04b/result.json", "../../12a/contract.json"],
        "limits": ["No routing trace was capturable, so the offline replay matrix could not be run; treated as no-change with that limitation."]},
    "22": {
        "plan": "11b", "bead": "FreeToken-mtp-1ll.22",
        "title": "Reduce expert transfer overhead on PCIe 3.0 - execution and acceptance",
        "measurements": {"copy_bw_gb_s": 12.2, "copy_mib": 25.3, "copy_ms": 2.18, "pcie_ceiling_gb_s_04b": 12.37, "h2d_total_gb_04b": 146.2},
        "analysis": ["The gather+copymissing path moves bytes at 12.2 GB/s, i.e. the measured PCIe 3.0 H2D ceiling (12.37) -> byte-bound.", "Call-count/coalescing changes cannot beat a bandwidth-bound copy; only fewer BYTES would help, which is the residency/cache lever (tasks 07/10), not the copy kernel."],
        "decision": "no_change: the copy path is already at the PCIe ceiling; no copy-kernel restructuring adopted.",
        "evidence": ["raw/bench-copy.txt", "../../04b/result.json"],
        "limits": ["Bench used the glm4.7-nvfp4 profile (small slots) as a generic copy-path measurement; per-call overhead at tiny sizes was not the metric."]},
    "26": {
        "plan": "13b", "bead": "FreeToken-mtp-1ll.26",
        "title": "PLE I/O overlap and thread-pool tuning - execution and acceptance",
        "measurements": {"target_test_ple_disk": "4 passed", "ple_share_of_tpot_12": 0.005, "rows_per_step": 16, "warm_row_us": 1.19},
        "analysis": ["PLE is <0.5% of TPOT (12): 16 rows x 1.19 us ~= 19 us vs ~46 ms/step = 0.04%.", "There is no headroom >=5% for overlap/thread tuning to win; the C++ extension could not add value here."],
        "decision": "no_change: keep the current PLE disk path; no overlap/thread-pool change adopted.",
        "evidence": ["raw/pytest-target.txt", "../../12a/result.json"],
        "limits": ["Workload-dependent PLE locality (12); no new measurement beyond the 12 analysis."]},
    "38": {
        "plan": "19b", "bead": "FreeToken-mtp-1ll.38",
        "title": "Prefill and multi-turn latency without hurting decode - execution and acceptance",
        "measurements": {"ttft_cold_ms": 6078.7, "ttft_radix_hit_ms": 5615.9, "ttft_turn2_ms": 5616.4,
                         "decode_tok_s": {"cold": 17.25, "radix_hit": 19.32, "turn2": 18.16}, "prompt_tokens": {"cold": 81, "turn2": 111}},
        "analysis": ["TTFT is ~5.6 s whether the prompt is cold or a radix prefix hit (6.08 vs 5.62 s): prefix reuse does NOT reduce TTFT, so the latency is dominated by the first prefill's per-layer expert movement, not token count.", "Decode is unaffected. No scheduler/pipeline change is justified by this probe."],
        "decision": "no_change: keep the current prefill scheduling; no TTFT improvement demonstrated.",
        "evidence": ["raw/ttft.json", "raw/run-bmisc.log", "../../../03a/contract.json"],
        "limits": ["One short prompt; TTFT floor is the expert-load cost. A longer-prompt prefill comparison was not run here."]},
}


def main() -> None:
    for d, t in RESULTS.items():
        out = {
            "task_key": t["plan"], "bead_id": t["bead"], "title": t["title"], "outcome": "no_change",
            "checkpoint_revision": "RadixArk/Qwen3.8-Flash-Next-NVFP4", "hardware_fingerprint_sha256": HW,
            "measurements": t["measurements"], "analysis": t["analysis"], "decision": t["decision"],
            "rollback_recipe": "n/a (no production change)",
            "limitations": t["limits"], "evidence": t["evidence"], "generated_utc": "2026-09-11",
        }
        p = BASE / d / "result.json"
        p.write_text(json.dumps(out, indent=2) + "\n")
        print("wrote", p)


main()
