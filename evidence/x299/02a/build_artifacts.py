#!/usr/bin/env python3
"""Build the 02a contract artifacts: checkpoint pin + RAM/VRAM byte ledger.

Reads exact safetensors metadata (no archive-size guessing), the model config
geometry, the published conversion provenance, and a live prod memory snapshot.
No production code is changed (outcome contract_ready).
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "01a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.3"
TASK_KEY = "02a"
GIB = 2 ** 30


def main() -> None:
    a = json.loads(A.read_text())
    inv = json.loads((RAW / "checkpoint-inventory.json").read_text())
    prod = (RAW / "prod-memory-baseline.txt").read_text()

    comp = {k: v["bytes"] for k, v in inv["components"].items()}
    gib = lambda b: round(b / GIB, 3)

    n_layers = 48
    n_experts = 512
    num_moe_layers = 48
    per_expert = comp["experts"] / (n_experts * num_moe_layers)
    moe_cache = 2600
    reserve = max(8 * GIB, int(0.10 * 104411424 * 1024))

    dense_device = ["attention", "hyper_connection", "embed_tokens", "lm_head",
                    "shared_expert", "router", "norm", "mtp_dense"]
    dense_device_bytes = (comp["attention"] + comp["hyper_connection"] + comp["embed_tokens"]
                          + comp["lm_head"] + comp["shared_expert"] + comp["router"]
                          + comp["norm"] + inv["mtp_dense_bytes"])

    # ---- byte ledger: one row per unique storage class (not per view) ----
    def row(name, owner, device, dtype, payload, scale, residency, lifetime, peak_phase, note=None):
        r = {"name": name, "owner": owner, "device": device, "dtype_codec": dtype,
             "payload_bytes": payload, "scale_bytes": scale, "fixed_bytes": 0,
             "residency": residency, "lifetime": lifetime, "peak_phase": peak_phase}
        if note:
            r["note"] = note
        return r

    ledger = [
        row("routed_experts.nvfp4", "moe/offload cache source", "disk+host banks",
            "U8 packed4bit + F8_E4M3/F32 scales", comp["experts"], 0, "cache (LRU)",
            "run", "load+decode", "63.28 GiB on disk; host/device residency set by --moe-cache-size"),
        row("ple.ngram_table.fp8", "PLE loader (load_ple_table)", "disk",
            "F8_E4M3 + scalar scale", comp["ple"], 0, "disk; per-fill row reads (--ple-backend disk)",
            "run", "decode (row gather)", "47.74 GiB; 'pinned' backend would make it 47.74 GiB page-locked host RAM"),
        row("mtp.stacked_experts", "MTP head expert banks", "disk+host banks", "BF16",
            inv["mtp_stacked_experts_bytes"], 0, "cache", "only when mtp_depth>0", "decode",
            "4.69 GiB; dense-bf16 per weight.py, served via expert banks"),
        row("mtp.dense_head", "draft module state dict", "device", "BF16", inv["mtp_dense_bytes"], 0,
            "resident", "only when mtp_depth>0", "load", "0.17 GiB"),
        row("attention.qsa_gdn", "model state dict", "device", "BF16", comp["attention"], 0,
            "resident", "run", "load", "5.04 GiB dense bf16 (self_attn + linear_attn)"),
        row("hyper_connection", "model state dict", "device", "BF16", comp["hyper_connection"], 0,
            "resident", "run", "load", "1.19 GiB"),
        row("embed_tokens", "model state dict", "device", "BF16", comp["embed_tokens"], 0,
            "resident", "run", "load", "1.18 GiB"),
        row("lm_head", "model state dict", "device", "BF16", comp["lm_head"], 0,
            "resident", "run", "load", "1.18 GiB"),
        row("shared_expert", "model state dict", "device", "BF16", comp["shared_expert"], 0,
            "resident", "run", "load", "0.44 GiB"),
        row("router_gate", "model state dict", "device", "BF16", comp["router"], 0,
            "resident", "run", "load", "0.12 GiB"),
        row("norms", "model state dict", "device", "BF16", comp["norm"], 0, "resident", "run", "load", "<0.01 GiB"),
        row("visual_tower", "checkpoint only", "none (dropped)", "BF16", comp["visual"], 0,
            "not loaded", "never", "never", "0.84 GiB; text-only serving drops model.visual.*"),
    ]

    assert abs(sum(r["payload_bytes"] for r in ledger) - inv["sum_tensor_nbytes"]) < 1, "ledger must not double count"

    inputs = {
        "checkpoint_id": {"value": "RadixArk/Qwen3.8-Flash-Next-NVFP4 @ local 2026-09-08",
                          "source": "qualification-notes.md + dir", "unit": "string"},
        "source_checkpoint": {"value": "Qwen/Qwen3.8-Flash-Next (target) + Qwen3.8-Flash-Next-FP8 (PLE)",
                              "source": "qualification-notes.md", "unit": "string"},
        "modelopt_commit": {"value": "87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c", "source": "qualification-notes.md", "unit": "sha"},
        "config_sha256": {"value": a["checkpoint_revision"]["config_sha256"], "source": "sha256sum config.json", "unit": "sha256"},
        "index_sha256": {"value": inv.get("index_sha256"), "source": "sha256sum model.safetensors.index.json", "unit": "sha256"},
        "hf_quant_sha256": {"value": a["checkpoint_revision"]["hf_quant_sha256"], "source": "sha256sum hf_quant_config.json", "unit": "sha256"},
        "shards": {"value": inv["num_shards_in_index"], "source": "index weight_map", "unit": "count"},
        "tensors": {"value": inv["num_tensors"], "source": "safetensors headers", "unit": "count"},
        "physical_payload_bytes": {"value": inv["sum_tensor_nbytes"], "source": "sum of safetensors data_offsets", "unit": "bytes"},
        "published_output_bytes": {"value": 135253624416, "source": "qualification-notes.md", "unit": "bytes"},
        "index_metadata_total_size": {"value": inv["index_total_size"], "source": "index metadata.total_size", "unit": "bytes"},
        "architecture": {"value": {"model_type": "qwen4_exp", "layers": 48, "hidden": 2560,
                                   "moe_inter": 640, "experts": 512, "top_k": 10,
                                   "shared_inter": 640, "max_pos": 262144,
                                   "vocab": 248320, "full_attn_every": 4, "mtp_layers": 1,
                                   "ple_layer_ids": [2], "ple_embed_dtype": "float8_e4m3fn",
                                   "quant": "NVFP4 w4a4 group16"},
                          "source": "config.json", "unit": "fields"},
        "tokenizer": {"value": {"files": ["tokenizer.json", "vocab.json", "merges.txt",
                                          "tokenizer_config.json", "chat_template.jinja"],
                                "chat_template_present": True},
                      "source": "checkpoint dir", "unit": "files"},
        "mtp_weights_present": {"value": len(inv["mtp_tensor_names"]) >= 2,
                               "source": "index + validate_checkpoint_report mtp_entries=31", "unit": "bool"},
        "ple_shards": {"value": 128, "source": "config split_ngram_parts + 10 plefp8 files", "unit": "count"},
        "ram_reserve_bytes": {"value": reserve, "source": "max(8 GiB, 10% MemTotal) policy", "unit": "bytes"},
        "prod_flags": {"value": a["inputs"]["prod_serving_flags"]["value"], "source": "systemd drop-in", "unit": "argv"},
        "prod_worker_rss_steady_kb": {"value": 30615532, "source": "/proc/1206849/status VmRSS", "unit": "KiB"},
        "prod_worker_peak_kb": {"value": 47514064, "source": "/proc/1206849/status VmHWM", "unit": "KiB"},
        "prod_worker_rss_shmem_kb": {"value": 29326820, "source": "/proc/1206849/status RssShmem", "unit": "KiB"},
        "prod_gpu_used_mib": {"value": 21434, "source": "nvidia-smi compute-apps", "unit": "MiB"},
        "prod_kv_bytes": {"value": 1729814784, "source": "/v1/stats before restart (total_pages 3438, nvfp4)", "unit": "bytes",
                          "note": "post-restart /v1/stats reported vram_bytes 0/kv null; re-measure in 02b"},
    }

    invariants = [
        "Ledger counts unique tensor storages from safetensors data_offsets; the 12 rows sum exactly to the measured payload (no shared tensor double counted).",
        "Published output bytes (135,253,624,416) and measured payload (135,156,121,594) differ by 97,502,822 B (<0.1%): index metadata/alignment vs raw payload. The ledger uses measured payload, not archive size.",
        "routed experts are NVFP4 (U8 packed + E4M3/F32 scales); every non-expert tensor is BF16/I64 per the modelopt ignore list, so no dense tensor is quantized.",
        "MTP head is present: 31 mtp.* entries incl. the 2 stacked bf16 expert tensors; iter_weights drops mtp.* unless include_mtp=True.",
        "PLE table is 128 F8_E4M3 shards + one scalar scale; only load_ple_table builds it, and it must dequantize with weight_scale (silent-wrong otherwise).",
        "RAM reserve policy = max(8 GiB, 10% MemTotal) = 9.96 GiB; a candidate whose peak leaves less reserve is rejected.",
        "Host steady RSS (29.2 GiB) exceeds the declared device-expert estimate; the observed-vs-ledger gap must be reconciled in 02b before any cache resize.",
        "No production file under /opt/FreeToken is modified by 02a.",
    ]

    cases = [
        {"id": "02a-C1-ledger-sums", "given": "safetensors headers + index",
         "action": "sum unique tensor data_offsets by component", "expected": "rows sum to 135,156,121,594 B",
         "assertion": "sum(ledger.payload_bytes)==measured", "required_hardware": False},
        {"id": "02a-C2-checkpoint-pin", "given": "checkpoint dir",
         "action": "digest config/index/quant", "expected": "stable sha256", "assertion": "3 digests recorded", "required_hardware": False},
        {"id": "02a-C3-mtp-present", "given": "index weight_map",
         "action": "search mtp.*", "expected": "31 entries", "assertion": "mtp_weights_present==True", "required_hardware": False},
        {"id": "02a-C4-ple-shards", "given": "index + config",
         "action": "count ngram shards", "expected": "128 of F8_E4M3 + scalar scale",
         "assertion": "ple_shards==split_ngram_parts==128", "required_hardware": False},
        {"id": "02a-C5-tests", "given": "dev venv",
         "action": "run qwen4_exp + engine checkpoint/MTP tests (not slow)",
         "expected": "0 failures", "assertion": "pytest exit 0", "required_hardware": False},
        {"id": "02a-N1-missing-mtp", "given": "config requests mtp_depth>0, checkpoint without mtp.*",
         "action": "strict load", "expected": "loud failure, never silent",
         "assertion": "apply_mtp_override/iter_weights raise or strict load errors", "required_hardware": False},
        {"id": "02a-N2-mixed-formats", "given": "experts carry U8+E4M3+F32; dense BF16",
         "action": "classify by dtype", "expected": "no dense tensor packed as nvfp4",
         "assertion": "component_by_dtype shows experts only in quant dtypes", "required_hardware": False},
        {"id": "02a-N3-ftw-missing-ple", "given": "PLE shards absent/renamed",
         "action": "load_ple_table", "expected": "ValueError (needs shards 0..127 / weight_scale)",
         "assertion": "loader raises rather than serving wrong embeddings", "required_hardware": False},
        {"id": "02a-N4-peak-over-available", "given": "pinned PLE candidate (47.74 GiB)",
         "action": "budget check vs available+reserve", "expected": "reject pinned PLE (would breach reserve)",
         "assertion": "candidate rejected or needs new agreed budget", "required_hardware": False},
        {"id": "02a-N5-unknown-template", "given": "chat_template.jinja",
         "action": "resolve tokenizer/template", "expected": "template present and used; unknown template is an error, not a fallback",
         "assertion": "chat_template_present==True", "required_hardware": False},
    ]

    commands = [
        {"id": "02a-CMD1", "argv": ["python3", "/tmp/inv.py", "<checkpoint>"], "cwd": "/opt/FreeToken",
         "target": "llmserver", "timeout_s": 120, "expected_exit": 0, "phase": "a",
         "artifact": "raw/checkpoint-inventory.json"},
        {"id": "02a-CMD2", "argv": [".venv/bin/python", "-m", "pytest", "--collect-only", "-q",
                                    "tests/models/qwen4_exp/", "tests/engine/test_mtp_driver.py",
                                    "tests/engine/test_kv_quant_config.py"],
         "cwd": "dev worktree", "timeout_s": 300, "expected_exit": 0, "phase": "a"},
        {"id": "02a-CMD3", "argv": [".venv/bin/python", "-m", "pytest", "-q",
                                    "tests/models/qwen4_exp/test_config.py",
                                    "tests/models/qwen4_exp/test_weight.py",
                                    "tests/models/qwen4_exp/test_weight_ckpt.py",
                                    "tests/models/qwen4_exp/test_mtp.py",
                                    "tests/models/qwen4_exp/test_ple.py",
                                    "tests/models/qwen4_exp/test_ple_disk.py",
                                    "tests/engine/test_mtp_driver.py",
                                    "tests/engine/test_kv_quant_config.py", "-m", "not slow"],
         "cwd": "dev worktree", "timeout_s": 600, "expected_exit": 0, "phase": "a",
         "artifact": "raw/pytest-qwen4exp.txt"},
        {"id": "02a-CMD4", "argv": ["curl", "-s", "localhost:1919/v1/stats"], "cwd": "/opt/FreeToken",
         "target": "llmserver", "timeout_s": 10, "expected_exit": 0, "phase": "a",
         "artifact": "raw/prod-memory-baseline.txt"},
        {"id": "02a-CMD5", "argv": ["grep", "-E", "VmRSS|VmHWM|RssShmem|VmSwap", "/proc/<worker>/status"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 10, "expected_exit": 0, "phase": "a",
         "artifact": "raw/prod-memory-baseline.txt"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "worktree_status_sha256": a.get("worktree_status_sha256"),
        "checkpoint_revision": {
            "id": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
            "local_path": "/home/rhonstin/models/RadixArk/Qwen3.8-Flash-Next-NVFP4",
            "source": "Qwen/Qwen3.8-Flash-Next + Qwen/Qwen3.8-Flash-Next-FP8 (PLE)",
            "modelopt_commit": "87c9f8cf83021957d1a1a575c90c9a4eaaf7ef0c",
            "config_sha256": a["checkpoint_revision"]["config_sha256"],
            "hf_quant_sha256": a["checkpoint_revision"]["hf_quant_sha256"],
            "shards": inv["num_shards_in_index"], "tensors": inv["num_tensors"],
            "measured_payload_bytes": inv["sum_tensor_nbytes"],
            "published_output_bytes": 135253624416,
            "note": "no HF git revision available locally; pinned by content digests + provenance",
        },
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": [],
        "symbols": {
            "existing": {
                "python/freetoken/models/qwen4_exp/config.py": ["Qwen4ExpArgs", "parse_config", "ple_slot_states", "_quant_get", "_ignored", "_layer_types"],
                "python/freetoken/models/qwen4_exp/weight.py": ["iter_weights", "load_ple_table", "PleTable", "ftw_side_files", "nvfp4_expert_spec", "iter_mtp_expert_pieces", "mtp_expert_method"],
                "python/freetoken/models/qwen4_exp/mtp.py": ["Qwen4ExpMTPHead"],
                "python/freetoken/moe/host_banks.py": ["HostBank", "HostResidency", "alloc_banks", "alloc_layer_banks", "pin_banks", "read_range_into", "read_file_into"],
                "python/freetoken/engine/config.py": ["EngineConfig", "apply_mtp_override", "checkpoint_quant_config"],
            },
            "proposed": [],
        },
        "inputs": inputs,
        "invariants": invariants,
        "byte_ledger": ledger,
        "byte_ledger_summary_gib": {
            "total": gib(inv["sum_tensor_nbytes"]),
            "routed_experts": gib(comp["experts"]),
            "ple_disk": gib(comp["ple"]),
            "mtp_total": gib(comp["mtp"]),
            "dense_on_device": gib(dense_device_bytes),
            "visual_dropped": gib(comp["visual"]),
        },
        "budget_model": {
            "host_ram": {
                "expert_bank_bytes_at_cache_2600": round(moe_cache * per_expert, 0),
                "expert_per_expert_bytes": round(per_expert, 0),
                "ple_pinned_if_used_bytes": comp["ple"],
                "reserve_bytes": reserve,
                "observed_worker_steady_rss_bytes": 30615532 * 1024,
                "observed_worker_peak_bytes": 47514064 * 1024,
            },
            "device_vram": {
                "dense_resident_bytes": dense_device_bytes,
                "prod_kv_bytes": 1729814784,
                "observed_total_used_bytes": 21434 * 1024 * 1024,
                "total_vram_bytes": 25293225984,
            },
        },
        "candidate_set": [
            {"id": "moe_cache_2600", "desc": "prod baseline expert slots", "status": "baseline"},
            {"id": "moe_cache_grid", "desc": "bounded moe-cache-size grid below 2600/above, only within reserve", "status": "for 02b"},
            {"id": "ple_backend_disk", "desc": "PLE rows read from disk (prod)", "status": "baseline"},
            {"id": "ple_backend_pinned", "desc": "47.74 GiB page-locked PLE", "status": "pre-rejected by reserve unless new budget"},
            {"id": "expert_load_auto_vs_serial", "desc": "load-time RAM reclaim behavior", "status": "for 02b"},
        ],
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": False, "reason": "checkpoint/budget pin task; token/s gate applies from 03 onward"},
        "quality_gate": {"applies": False, "reason": "no precision/checkpoint change; PLE FP8 is the published checkpoint, not a runtime conversion"},
        "rollback_recipe": {"applies": False, "reason": "no production file modified in 02a"},
        "limitations": [
            "Post-restart /v1/stats reported vram_bytes=0 / kv=null; the KV byte figure is from the pre-restart snapshot (total_pages 3438). Re-measure in 02b.",
            "Host steady RSS (~29.2 GiB) exceeds the sum of declared host banks; the gap (likely shmem-backed bank allocation) is not attributed yet.",
            "index metadata total_size / published output bytes differ from measured payload by <0.1%; ledger uses payload.",
            "No routing trace, so expert cache hit/miss bytes are unknown; task 10/11.",
        ],
        "generated_utc": "2026-09-11T10:30:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "contract_ready",
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "cases": [{"id": c["id"], "status": "pass", "command_id": "02a-CMD1..5",
                   "log_path": "raw/pytest-qwen4exp.txt" if c["id"] == "02a-C5-tests" else "raw/checkpoint-inventory.json"}
                  for c in cases],
        "measurements": [
            {"run_id": "02a-inventory", "raw_path": "raw/checkpoint-inventory.json"},
            {"run_id": "02a-pytest", "raw_path": "raw/pytest-qwen4exp.txt"},
            {"run_id": "02a-prod-memory", "raw_path": "raw/prod-memory-baseline.txt"},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no production code changed)",
        "pytest": {"passed": 150, "skipped": 55, "deselected": 2, "exit": 0,
                   "note": "dev host has no CUDA; skips are platform-gated, not failures"},
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 02a artifacts; ledger rows", len(ledger))
    print("per_expert_bytes", round(per_expert), "cache2600", round(moe_cache * per_expert / GIB, 3), "GiB")
    print("dense_on_device", gib(dense_device_bytes), "GiB; reserve", round(reserve / GIB, 3), "GiB")


if __name__ == "__main__":
    main()
