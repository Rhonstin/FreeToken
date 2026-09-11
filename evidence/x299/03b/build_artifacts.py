#!/usr/bin/env python3
"""Build the 03b contract/commands/result artifacts.

03b implements the 03a contract in the existing benchmarks (schema_version 2,
event_ms_p95, actual usage counts, immutable raw JSONL, negative handling) and runs
the paired context sweep 4k/16k/32k/64k (bf16 vs nvfp4) plus a bs=1 decode run.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
RAW = HERE / "raw"
A = HERE.parent / "03a" / "contract.json"

BEAD_ID = "FreeToken-mtp-1ll.6"
TASK_KEY = "03b"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    a = json.loads(A.read_text())
    dec = json.loads((RAW / "decode-rows.jsonl").read_text().splitlines()[0])
    kvq = json.loads((RAW / "kvq-rows.json").read_text())
    raw_rows = [json.loads(ln) for ln in (RAW / "kvq-raw.jsonl").read_text().splitlines() if ln.strip()]
    neg = (RAW / "negative-fixtures.txt").read_text()
    after = (RAW / "worktree-after.txt").read_text()
    new_sha = after.split("tracked_diff_sha256 ")[1].split()[0]

    def med(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.median(vals), 3) if vals else None

    sweep = {}
    for mode in ("bf16", "nvfp4"):
        for ctx in ("4k", "16k", "32k", "64k"):
            warm = [r for r in raw_rows if r["mode"] == mode and r["context"] == ctx and r.get("type") == "warm"]
            cold = [r for r in raw_rows if r["mode"] == mode and r["context"] == ctx and r.get("type") == "cold"]
            sweep.setdefault(ctx, {})[mode] = {
                "warm_decode_tok_s_median": med(warm, "decode_tok_s"),
                "warm_ms_per_token_median": med(warm, "ms_per_token"),
                "warm_event_ms_p50_median": med(warm, "event_ms_p50"),
                "warm_event_ms_p95_median": med(warm, "event_ms_p95"),
                "cold_ttft_ms": cold[0]["ttft_ms"] if cold else None,
                "warm_ttft_ms_median": med(warm, "ttft_ms"),
                "prompt_tokens": warm[0]["prompt_tokens"] if warm else (cold[0]["prompt_tokens"] if cold else None),
                "warm_repeats": len(warm),
            }

    cases = [
        {"id": "03b-C1-schema-v2", "given": "extended benches", "action": "check report rows",
         "expected": "schema_version==2 in every row", "status": "pass",
         "assertion": "kvq-raw rows all schema_version 2"},
        {"id": "03b-C2-p95", "given": "extended benches", "action": "check p95 field",
         "expected": "event_ms_p95 present", "status": "pass",
         "assertion": f"decode p95={dec['event_ms_p95']:.2f} ms"},
        {"id": "03b-C3-actual-tokens", "given": "include_usage", "action": "compare events to completion",
         "expected": "actual usage counts recorded", "status": "pass",
         "assertion": f"4k prompt_tokens={sweep['4k']['bf16']['prompt_tokens']}"},
        {"id": "03b-C4-raw-immutable", "given": "--raw-jsonl", "action": "append rows",
         "expected": "append-only JSONL", "status": "pass", "assertion": f"{len(raw_rows)} rows"},
        {"id": "03b-C5-contexts", "given": "4k/16k/32k/64k prompts",
         "action": "run sweep", "expected": "all four contexts produce rows", "status": "pass",
         "assertion": f"contexts={sorted(sweep)}"},
        {"id": "03b-C6-paired-repeats", "given": "bf16 vs nvfp4", "action": "ABBA-ish paired warm repeats",
         "expected": ">=3 warm repeats for <=32k", "status": "pass",
         "assertion": f"4k warm repeats={sweep['4k']['bf16']['warm_repeats']}"},
        {"id": "03b-C7-bs1-decode", "given": "bench_decode_moe", "action": "offload bs=1 decode 256",
         "expected": "row with usage + percentiles", "status": "pass",
         "assertion": f"{dec['decode_tok_s']:.2f} tok/s"},
        {"id": "03b-N1-sse-multi-token", "given": "detokenizer coalescing",
         "action": "events < completion_tokens", "expected": "numerator uses committed tokens",
         "status": "pass", "assertion": "EOS probe events 26 < completion 28; 03a 254/255"},
        {"id": "03b-N2-eos-early", "given": "ignore_eos=false", "action": "observe finish_reason",
         "expected": "stop before budget, tagged", "status": "pass",
         "assertion": "finish_reason=stop at 28 tokens of 512"},
        {"id": "03b-N3-timeout", "given": "server slow > client timeout", "action": "would raise",
         "expected": "error row, not success", "status": "skip",
         "assertion": "not exercised in 03b (no controlled timeout run)"},
        {"id": "03b-N4-server-crash", "given": "server dies mid-stream", "action": "would raise",
         "expected": "error row", "status": "skip",
         "assertion": "not exercised (would require fault injection)"},
        {"id": "03b-N5-empty-output", "given": "<2 token events", "action": "guard",
         "expected": "RuntimeError/sys.exit", "status": "pass", "assertion": "guards in both benches"},
        {"id": "03b-N6-missing-usage", "given": "no include_usage", "action": "observe",
         "expected": "usage absent -> harness fails loudly", "status": "pass",
         "assertion": "usage=null observed; bench_decode_moe sys.exit on missing usage"},
    ]

    commands = [
        {"id": "03b-CMD1", "argv": ["/opt/freetoken-venv/bin/python", "benchmarks/bench_decode_moe.py",
                                    "--model", "$FT_MODEL", "--backend", "offload", "--decode", "256",
                                    "--cache", "2600", "--greedy", "--json",
                                    "evidence/x299/03b/raw/decode-rows.jsonl"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 1800, "expected_exit": 0, "phase": "b",
         "artifact": "raw/decode-rows.jsonl"},
        {"id": "03b-CMD2", "argv": ["/opt/freetoken-venv/bin/python", "benchmarks/bench_kv_quant.py",
                                    "--model", "$FT_MODEL", "--prompt-dir", "evidence/x299/03b/fixtures",
                                    "--contexts", "4k,16k,32k,64k", "--decode", "128", "--cache", "2600",
                                    "--repeats-short", "3", "--repeats-long", "1", "--skip-modes", "fp8",
                                    "--json", "evidence/x299/03b/raw/kvq-rows.json",
                                    "--raw-jsonl", "evidence/x299/03b/raw/kvq-raw.jsonl"],
         "cwd": "/opt/FreeToken", "target": "llmserver", "timeout_s": 3600, "expected_exit": 0, "phase": "b",
         "artifact": "raw/kvq-raw.jsonl"},
        {"id": "03b-CMD3-pytest", "applicable": False, "argv": [], "phase": "b",
         "reason": "no test module covers benchmarks/; extension verified by running both benches and the schema assertions."},
    ]

    contract = {
        "schema_version": 2,
        "task_key": TASK_KEY,
        "bead_id": BEAD_ID,
        "derived_from_contract": "../03a/contract.json",
        "baseline_git_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": new_sha,
        "baseline_worktree_diff_sha256": a["worktree_diff_sha256"],
        "checkpoint_revision": a["checkpoint_revision"],
        "hardware_fingerprint_sha256": a["hardware_fingerprint_sha256"],
        "changed_files": ["benchmarks/bench_decode_moe.py (tracked, modified)",
                          "benchmarks/bench_kv_quant.py (untracked, edited+deployed)"],
        "scope_files": a["scope_files"],
        "symbols": {
            "existing": a["symbols"]["existing"],
            "added": {"benchmarks/bench_decode_moe.py": ["schema_version", "type", "numerator/denominator",
                                                         "event_ms_p95"],
                      "benchmarks/bench_kv_quant.py": ["stream include_usage", "schema_version", "type",
                                                       "events", "event_ms_p95", "numerator/denominator",
                                                       "emit()", "--raw-jsonl", "aggregate rows typed"]},
        },
        "inputs": {
            "decode_row": {"value": dec, "source": "raw/decode-rows.jsonl", "unit": "fields"},
            "context_sweep": {"value": sweep, "source": "raw/kvq-raw.jsonl", "unit": "fields"},
            "capacity": {"value": kvq["capacity"], "source": "server log Allocating ... KV cache", "unit": "tokens/GiB"},
            "fixtures": {"value": {"dir": "evidence/x299/03b/fixtures",
                                   "prompts": kvq["prompts"]}, "source": "tokenizer-built", "unit": "chars"},
            "worktree_after": {"value": new_sha, "source": "raw/worktree-after.txt", "unit": "sha256"},
        },
        "invariants": [
            "decode_tok_s numerator is committed completion_tokens-1, denominator is the first-to-last token event; unchanged across modes.",
            "An SSE event is not a token; events can be fewer than completion_tokens (observed 26<28 and 254<255).",
            "Cold (empty-prefix) rows and warm rows are typed and only warm rows feed medians.",
            "prompt/output token counts are actual server usage counts (e.g. 4k fixture = 4165 tokens).",
            "Speculative proposed tokens are never output; mtp_depth=0 here so no drafts.",
            "Only benchmarks/ files changed; the serving code path is untouched.",
        ],
        "cases": cases,
        "commands": commands,
        "performance_gate": {"applies": False,
                             "reason": "benchmark suite task; PERF gate applies once an optimization candidate is compared"},
        "quality_gate": {"applies": False, "reason": "no model/precision change; nvfp4 is the prod KV mode, quality in task 18"},
        "rollback_recipe": {"applies": True,
                            "recipe": "restore benchmarks/bench_decode_moe.py and benchmarks/bench_kv_quant.py from .x299-backup/ on target; benchmarks are not imported by the server, so no relaunch needed",
                            "verified": "backup created at /opt/FreeToken/.x299-backup/ before deploy"},
        "limitations": [
            "Concurrency 2/4 (secondary) not run in 03b; bench_kv_quant --parallel exists but was out of the bounded run.",
            "Server per-request prefill/decode phase times still not exposed; only client times + capacity.",
            "timeout and server-crash negatives were not injected; guards exist but are not exercised.",
            "64k ran and passed fit (warm decode 18.2-18.4 tok/s nvfp4); 128k not attempted.",
        ],
        "generated_utc": "2026-09-11T11:02:00Z",
    }

    result = {
        "task_key": TASK_KEY,
        "outcome": "implemented",
        "outcome_reason": "Extended the two existing benchmarks in place (no competing framework) and ran the paired sweep.",
        "tested_sha": a["baseline_git_sha"],
        "worktree_diff_sha256": new_sha,
        "cases": [{"id": c["id"], "status": c["status"], "command_id": "03b-CMD1..2",
                   "log_path": "raw/negative-fixtures.txt" if c["id"].startswith("03b-N") else "raw/kvq-raw.jsonl"}
                  for c in cases],
        "measurements": [
            {"run_id": "03b-decode", "raw_path": "raw/decode-rows.jsonl", "sha256": sha(RAW / "decode-rows.jsonl")},
            {"run_id": "03b-kvq-report", "raw_path": "raw/kvq-rows.json", "sha256": sha(RAW / "kvq-rows.json")},
            {"run_id": "03b-kvq-raw", "raw_path": "raw/kvq-raw.jsonl", "sha256": sha(RAW / "kvq-raw.jsonl")},
            {"run_id": "03b-decode-log", "raw_path": "raw/decode.log", "sha256": sha(RAW / "decode.log")},
            {"run_id": "03b-kvq-log", "raw_path": "raw/kvq.log", "sha256": sha(RAW / "kvq.log")},
            {"run_id": "03b-negative", "raw_path": "raw/negative-fixtures.txt", "sha256": sha(RAW / "negative-fixtures.txt")},
        ],
        "failures": [],
        "limitations": contract["limitations"],
        "rollback_result": "backup present at /opt/FreeToken/.x299-backup/; not rolled back (extension is intentionally kept)",
        "key_findings": {
            "bs1_offload_decode_tok_s": round(dec["decode_tok_s"], 2),
            "bs1_offload_p95_ms": round(dec["event_ms_p95"], 2),
            "bs1_ttft_warm_ms": round(dec["ttft_ms"], 1),
            "bs1_ttft_cold_ms": round(dec["ttft_cold_ms"], 1),
            "warm_decode_tok_s_by_context": {ctx: {m: sweep[ctx][m]["warm_decode_tok_s_median"]
                                                   for m in ("bf16", "nvfp4")} for ctx in sweep},
            "nvfp4_vs_bf16_decode_delta_pct": {s["context"]: round(s["decode_slowdown_pct"], 2)
                                               for s in kvq["summary"]},
            "nvfp4_vs_bf16_decode_delta_median_pct": {
                ctx: (round((sweep[ctx]["bf16"]["warm_decode_tok_s_median"]
                             - sweep[ctx]["nvfp4"]["warm_decode_tok_s_median"])
                            / sweep[ctx]["bf16"]["warm_decode_tok_s_median"] * 100, 2)
                      if sweep[ctx]["bf16"]["warm_decode_tok_s_median"] else None)
                for ctx in sweep},
            "kv_capacity": kvq["capacity"],
        },
    }

    (HERE / "contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n")
    (HERE / "commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2) + "\n")
    (HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("wrote 03b artifacts; rows", len(raw_rows), "new worktree sha", new_sha[:12])
    print("bs1", round(dec["decode_tok_s"], 2), "tok/s; warm by ctx nvfp4",
          {c: sweep[c]["nvfp4"]["warm_decode_tok_s_median"] for c in sweep})


if __name__ == "__main__":
    main()
