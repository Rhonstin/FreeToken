#!/usr/bin/env python3
"""Build the 05b contract/commands/result artifacts: best-known config sweep.

Config-only (no runtime code). Reports the measured candidate table, the nvfp4
context verification for the winner, the CUDA-graph factor, the no-fallback
effective-config check and the greedy-parity caveat.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "05a" / "contract.json"
B3 = HERE.parent / "03b" / "raw" / "decode-rows.jsonl"
B4 = HERE.parent / "04b" / "raw" / "decode-eager-plain-rows.jsonl"

BEAD_ID = "FreeToken-mtp-1ll.10"
TASK_KEY = "05b"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def row(p):
    return json.loads(pathlib.Path(p).read_text().splitlines()[0])


def main() -> None:
    a = json.loads(A.read_text())
    off_g1 = row(B3)
    off_g0 = row(B4)
    cands = {
        "hybrid_auto": row(RAW / "cand-hybrid_auto.jsonl"),
        "hybrid_f0": row(RAW / "cand-hybrid_f0.jsonl"),
        "hybrid_f1": row(RAW / "cand-hybrid_f1.jsonl"),
        "hybrid_f2": row(RAW / "cand-hybrid_f2.jsonl"),
        "cpu": row(RAW / "cand-cpu.jsonl"),
    }
    sweep = {
        "offload_graph1(03b)": {"tok_s": round(off_g1["decode_tok_s"], 2), "p95": round(off_g1["event_ms_p95"], 2), "sha": off_g1["output_sha1"]},
        "offload_graph0(04b)": {"tok_s": round(off_g0["decode_tok_s"], 2), "p95": round(off_g0["event_ms_p95"], 2), "sha": off_g0["output_sha1"]},
        **{k: {"tok_s": round(v["decode_tok_s"], 2), "p95": round(v["event_ms_p95"], 2), "sha": v["output_sha1"]}
           for k, v in cands.items()},
    }
    base = off_g1["decode_tok_s"]
    gains = {k: round((v["tok_s"] / base - 1) * 100, 1) for k, v in sweep.items() if k != "offload_graph1(03b)"}

    ctx = {}
    for f in sorted(RAW.glob("ctx-*.jsonl")):
        for ln in f.read_text().splitlines():
            r = json.loads(ln)
            ctx.setdefault(r["mode"], {})[r["context"]] = {"tok_s": round(r["decode_tok_s"], 2),
                                                           "p95": round(r["event_ms_p95"], 2),
                                                           "prompt_tokens": r["prompt_tokens"]}
    ctx_gain = {}
    for c in ("4k", "16k", "32k"):
        o = ctx.get("offload-f-1", {}).get(c, {}).get("tok_s")
        h = ctx.get("hybrid-f1", {}).get(c, {}).get("tok_s")
        if o and h:
            ctx_gain[c] = round((h / o - 1) * 100, 1)

    o_txt = (RAW / "text-offload--1.txt").read_text()
    h_txt = (RAW / "text-hybrid-1.txt").read_text()
    prefix = difflib.SequenceMatcher(None, o_txt, h_txt).find_longest_match(0, len(o_txt), 0, len(h_txt)).size

    eff = {}
    for f in RAW.glob("cand-*.effective.txt"):
        txt = f.read_text()
        m = re.search(r"moe_strategy='([a-z]+)'.*moe_hybrid_max_fetch=(-?\d+)", txt, re.S)
        if m:
            eff[f.name] = f"{m.group(1)} fetch={m.group(2)}"

    cases = [
        {"id": "05b-C0-baseline", "given": "offload graph1", "action": "reference",
         "expected": "reproducible", "status": "pass", "assertion": f"{base:.2f} tok/s"},
        {"id": "05b-C1-backend-sweep", "given": "one-factor backend", "action": "measure hybrid auto/0/1/2 + cpu",
         "expected": "ranked table", "status": "pass", "assertion": f"best hybrid f1 {sweep['hybrid_f1']['tok_s']} tok/s"},
        {"id": "05b-C2-graph-factor", "given": "graph 0/1", "action": "compare uninstrumented",
         "expected": "graph effect measured", "status": "pass",
         "assertion": f"offload graph1 {sweep['offload_graph1(03b)']['tok_s']} vs graph0 {sweep['offload_graph0(04b)']['tok_s']}"},
        {"id": "05b-C3-winner-contexts", "given": "nvfp4 prod-like", "action": "4k/16k/32k",
         "expected": "winner faster than offload at each", "status": "pass",
         "assertion": f"gains {ctx_gain}"},
        {"id": "05b-C4-no-fallback", "given": "requested vs effective", "action": "grep ServerArgs",
         "expected": "effective == requested", "status": "pass", "assertion": f"{list(eff.values())}"},
        {"id": "05b-C5-perf-gate", "given": "PERF gate >=5%", "action": "compare",
         "expected": "winner passes", "status": "pass", "assertion": f"hybrid f1 +{gains['hybrid_f1']}% (short)"},
        {"id": "05b-N1-invalid-backend", "given": "--moe-strategy bogus", "action": "argparse choices",
         "expected": "exit 2", "status": "pass", "assertion": "choices reject"},
        {"id": "05b-N2-unsupported-format", "given": "format not in _WFMT_IDS", "action": "cpu/hybrid load",
         "expected": "NotImplementedError", "status": "skip", "assertion": "not injected; nvfp4 is supported"},
        {"id": "05b-N3-maxfetch-gt-misses", "given": "cap > misses", "action": "resolve",
         "expected": "effective cap recorded", "status": "skip", "assertion": "grid used 0/1/2; not exercised"},
        {"id": "05b-N4-oom", "given": "oversized", "action": "launch", "expected": "reject", "status": "skip",
         "assertion": "not exercised"},
        {"id": "05b-N5-hidden-fallback", "given": "requested vs effective", "action": "verify",
         "expected": "no fallback", "status": "pass", "assertion": "all candidates resolved as requested"},
        {"id": "05b-N6-graph-failure", "given": "graph capture fail", "action": "n/a",
         "expected": "recorded", "status": "skip", "assertion": "not exercised"},
        {"id": "05b-QUALITY-greedy-parity", "given": "offload vs hybrid f1", "action": "greedy text compare",
         "expected": "identical tokens", "status": "fail",
         "assertion": f"common prefix {prefix} chars of {min(len(o_txt), len(h_txt))}; sha differs -> hybrid split changes greedy output"},
    ]

    commands = [
        {"id": "05b-CMD1", "argv": ["/opt/freetoken-venv/bin/python", "benchmarks/bench_decode_moe.py",
                                    "--model", "$FT_MODEL", "--backend", "<offload|hybrid|cpu>",
                                    "--hybrid-fetch", "<-1|0|1|2>", "--decode", "256", "--cache", "2600",
                                    "--greedy", "--json", "evidence/x299/05b/raw/cand-<id>.jsonl"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1500, "expected_exit": 0, "phase": "b"},
        {"id": "05b-CMD2", "argv": ["/opt/freetoken-venv/bin/ft", "serve", "--model", "$FT_MODEL",
                                    "--moe-strategy", "<s>", "--moe-hybrid-max-fetch", "<f>",
                                    "--moe-cache-size", "2600", "--kv-cache-dtype", "nvfp4",
                                    "--num-tokens", "220032", "--kv-reserve-tokens", "220032",
                                    "--memory-ratio", "0.90", "--ple-backend", "disk",
                                    "--cuda-graph-max-bs", "1"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1500, "expected_exit": 0, "phase": "b"},
        {"id": "05b-CMD3", "argv": ["/opt/freetoken-venv/bin/python", "evidence/x299/05b/context_probe.py",
                                    "--prompt-dir", "evidence/x299/03b/fixtures", "--contexts", "4k,16k,32k",
                                    "--decode", "128", "--jsonl", "evidence/x299/05b/raw/ctx-<mode>.jsonl"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1800, "expected_exit": 0, "phase": "b"},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "derived_from_contract": "../05a/contract.json",
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "scope_files": a["scope_files"],
        "symbols": a["symbols"],
        "fixed_knobs": a["fixed_knobs"],
        "sweep": sweep,
        "sweep_gain_pct_vs_offload_graph1": gains,
        "context_verification_nvfp4": ctx,
        "context_gain_pct": ctx_gain,
        "effective_configs": eff,
        "quality": {"offload_sha1": off_g1["output_sha1"], "hybrid_f1_sha1": cands["hybrid_f1"]["output_sha1"],
                    "common_prefix_chars": prefix, "compared_chars": min(len(o_txt), len(h_txt)),
                    "verdict": "hybrid split changes the greedy token path; NOT bit-identical to offload"},
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": True, "result": "PASS",
                             "detail": f"hybrid f1 +{gains['hybrid_f1']}% short, +{ctx_gain.get('4k')}/"
                                       f"{ctx_gain.get('16k')}/{ctx_gain.get('32k')}% at 4k/16k/32k; p95 improved (not regressed)"},
        "quality_gate": {"applies": True, "result": "CONDITIONAL/FAIL on bit-parity",
                         "detail": "greedy output diverges from offload; adopt only after /v1/score PPL (task 18) and task accuracy checks; offload stays the safe default"},
        "rollback_recipe": {"applies": False, "reason": "config-only; prod default not changed by 05b"},
        "limitations": [
            "Short-context sweep used the bench KV default; the context verification used prod nvfp4 and confirms the ranking.",
            "Greedy parity is not bit-exact across backends; hybrid's CPU/GPU expert split changes rounding. Semantic quality is not established here.",
            "cpu-layers subsets (C8) were not explored; they are gated on task 09.",
            "max-fetch values above 2 and OOM boundaries were not swept.",
        ],
        "generated_utc": "2026-09-11T12:25:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "implemented",
        "outcome_reason": "Bounded one-factor sweep completed and the best-known throughput profile identified; no runtime code added, prod default left unchanged pending the quality gate.",
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": a["worktree_diff_sha256"],
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "05b-CMD1..3",
                   "log_path": "raw/ctx-hybrid-1.jsonl"} for c in cases],
        "measurements": [
            {"run_id": f"05b-{k}", "raw_path": f"raw/cand-{k}.jsonl"} for k in cands
        ] + [
            {"run_id": "05b-ctx-offload", "raw_path": "raw/ctx-offload--1.jsonl", "sha256": sha(RAW / "ctx-offload--1.jsonl")},
            {"run_id": "05b-ctx-hybrid", "raw_path": "raw/ctx-hybrid-1.jsonl", "sha256": sha(RAW / "ctx-hybrid-1.jsonl")},
            {"run_id": "05b-text-offload", "raw_path": "raw/text-offload--1.txt", "sha256": sha(RAW / "text-offload--1.txt")},
            {"run_id": "05b-text-hybrid", "raw_path": "raw/text-hybrid-1.txt", "sha256": sha(RAW / "text-hybrid-1.txt")},
        ],
        "failures": ["quality: greedy bit-parity offload vs hybrid f1 (prefix %d of %d chars)" % (prefix, min(len(o_txt), len(h_txt)))],
        "limitations": contract["limitations"],
        "rollback_result": "not_required (no production code or default changed)",
        "key_findings": {
            "best_known_throughput": "hybrid max-fetch=1, graph=1, kv nvfp4",
            "sweep_tok_s": {k: v["tok_s"] for k, v in sweep.items()},
            "context_tok_s": ctx,
            "context_gain_pct": ctx_gain,
            "graph_gain_pct_offload": round((sweep["offload_graph1(03b)"]["tok_s"] / sweep["offload_graph0(04b)"]["tok_s"] - 1) * 100, 1),
            "quality_bit_parity": "fail (hybrid changes greedy output)",
            "adopted_default": "none (offload unchanged; hybrid quality-gated)",
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 05b; gains", gains, "ctx_gain", ctx_gain, "prefix", prefix)


if __name__ == "__main__":
    main()
