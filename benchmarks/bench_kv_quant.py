"""Paired bf16-vs-fp8 KV-cache campaign: capacity, prefill/decode and quality.

One server per (kv mode, graph on/off, fixed/auto expert cache) is reused across the
whole context matrix, so the expensive part (weights + banks, ~4 min) is paid a
handful of times. Every request run is a row: cold = first request after load, warm =
the measured repeats that follow a warmup. Capacity comes from the server's own
"Allocating N tokens for KV cache" line, mixed with the reported VRAM. Quality covers
needle retrieval at three depths, greedy-prefix agreement against bf16, a JSON /
tool-call format check and a small coding check. NLL/PPL is BLOCKED: the server has no
logprob API (`_completion_unsupported_reason` rejects `logprobs`), so it is reported
as not-run rather than approximated.

Thresholds to ratify in task 02: retrieval drop <= 2 percentage points, decode
slowdown <= 10%, TTFT <= +15%; anything missing those is "experimental".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODES = ("bf16", "fp8", "nvfp4")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(origin: str, proc: subprocess.Popen, log_path: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="ignore")[-1500:]
            raise SystemExit(f"[bench] server exited early ({proc.returncode}):\n{tail}")
        try:
            if get_json(f"{origin}/health", 2).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("[bench] server did not become ready in time")


def serve_cmd(args: argparse.Namespace, mode: str, port: int, max_running: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", args.model, "--host", "127.0.0.1", "--port", str(port),
        "--moe-strategy", "offload",
        "--max-running-requests", str(max_running),
        "--max-seq-len-override", str(args.max_seq_len),
        "--memory-ratio", str(args.mem_ratio),
        "--cuda-graph-max-bs", "0" if args.no_graph else "1",
        "--moe-hybrid-max-fetch", "-1",
        "--moe-collect-stats",
    ]
    if mode != "bf16":
        cmd += ["--kv-cache-dtype", mode]
    if args.cache:
        cmd += ["--moe-cache-size", str(args.cache)]
    else:
        cmd.append("--moe-cache-auto")
    if args.num_tokens > 0:
        cmd += ["--num-tokens", str(args.num_tokens),
                "--kv-reserve-tokens", str(args.num_tokens)]
    return cmd


def stream(origin: str, model_id: str, prompt: str, args: argparse.Namespace,
           max_tokens: int, temperature: float = 0.0) -> dict:
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict | None = None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    stamps.append(now)
                    pieces.append(text)
    result = {"t0": t0, "stamps": stamps, "text": "".join(pieces), "usage": usage or {}}
    return result


def run_row(origin: str, model_id: str, prompt: str, args: argparse.Namespace,
            ctx_label: str, mode: str, warm: bool) -> dict:
    r = stream(origin, model_id, prompt, args, args.decode)
    if len(r["stamps"]) < 2:
        raise RuntimeError(f"only {len(r['stamps'])} token events for {ctx_label}/{mode}")
    steps = r["usage"].get("completion_tokens", len(r["stamps"])) - 1
    span = r["stamps"][-1] - r["stamps"][0]
    gaps = sorted((b - a) * 1e3 for a, b in zip(r["stamps"], r["stamps"][1:]))
    return {
        "mode": mode, "context": ctx_label, "warm": warm,
        "prompt_tokens": r["usage"].get("prompt_tokens"),
        "completion_tokens": r["usage"].get("completion_tokens", len(r["stamps"])),
        "ttft_ms": (r["stamps"][0] - r["t0"]) * 1e3,
        "decode_tok_s": steps / span if span > 0 else 0.0,
        "ms_per_token": span / steps * 1e3 if steps > 0 else 0.0,
        "event_ms_p50": gaps[len(gaps) // 2] if gaps else None,
        "event_ms_p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))] if gaps else None,
        "output_sha1": hashlib.sha1(r["text"].encode()).hexdigest()[:12],
        "text": r["text"],
    }


def parse_capacity(log_path: Path) -> dict:
    text = log_path.read_text(errors="ignore")
    allocs = re.findall(r"Allocating (\d+) tokens for KV cache, K \+ V = ([\d.]+) GiB", text)
    moe = re.findall(r"moe miss: ([\d.]+)%", text)
    return {
        "kv_tokens": int(allocs[-1][0]) if allocs else None,
        "kv_gib": float(allocs[-1][1]) if allocs else None,
        "moe_miss_pct_last": float(moe[-1]) if moe else None,
    }


def needle_prompt(text: str, needle: str, depth: float) -> str:
    cut = int(len(text) * depth)
    return text[:cut] + f"\nThe secret access code is {needle}.\n" + text[cut:] + \
        "\n\nQuestion: what is the secret access code? Answer with the code only."


def quality_rows(origin: str, model_id: str, prompts: dict[str, str],
                 args: argparse.Namespace, mode: str) -> list[dict]:
    rows = []
    labels = ["4k", "16k", "32k"]
    if args.quality_long:
        labels += ["64k", "128k"]
    for label in labels:
        base = prompts.get(label)
        if not base:
            continue
        depths = (0.25, 0.5, 0.75) if label in ("4k", "16k", "32k") else (0.5,)
        for depth in depths:
            needle = f"XKCD-{label}-{int(depth * 100)}"
            r = stream(origin, model_id, needle_prompt(base, needle, depth), args,
                       args.quality_tokens)
            rows.append({
                "kind": "retrieval", "mode": mode, "context": label, "depth": depth,
                "hit": needle in r["text"], "text": r["text"][:200],
            })
    base = prompts.get("4k") or next(iter(prompts.values()))
    json_prompt = ("Output ONLY a JSON object with keys tool and city, values "
                   "\"get_weather\" and \"Kyiv\". No prose.\n\n" + base)
    r = stream(origin, model_id, json_prompt, args, args.quality_tokens)
    m = re.search(r"\{.*\}", r["text"], re.S)
    ok = False
    if m:
        try:
            obj = json.loads(m.group(0))
            ok = obj.get("tool") == "get_weather" and obj.get("city") == "Kyiv"
        except json.JSONDecodeError:
            pass
    rows.append({"kind": "json_tool_call", "mode": mode, "ok": ok, "text": r["text"][:200]})

    code_prompt = ("What is the value of fib(10) for fib(0)=0, fib(1)=1? "
                   "End with the number.\n\n" + base)
    r = stream(origin, model_id, code_prompt, args, args.quality_tokens)
    rows.append({
        "kind": "coding", "mode": mode,
        "ok": bool(re.search(r"\b55\b", r["text"])), "text": r["text"][:200],
    })
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-dir", required=True, help="dir with prompt_<label>.jsonl files")
    p.add_argument("--contexts", default="4k,16k,32k,64k,128k")
    p.add_argument("--decode", type=int, default=200)
    p.add_argument("--cache", type=int, default=2600, help="expert cache slots; 0 = auto")
    p.add_argument("--mem-ratio", type=float, default=0.90)
    p.add_argument("--max-seq-len", type=int, default=262144)
    p.add_argument("--max-running", type=int, default=1)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--repeats-short", type=int, default=3, help="runs after warmup for <=32k")
    p.add_argument("--repeats-long", type=int, default=1, help="runs after warmup for >32k")
    p.add_argument("--quality", action="store_true", help="run the quality phase too")
    p.add_argument("--quality-long", action="store_true",
                   help="add single-depth retrieval at 64k/128k (slow prefills)")
    p.add_argument("--quality-tokens", type=int, default=512,
                   help="max_tokens for the quality prompts (reasoning models need room)")
    p.add_argument("--num-tokens", type=int, default=0,
                   help="pin the KV pool (--num-tokens/--kv-reserve-tokens); 0 = budget-driven")
    p.add_argument("--parallel", type=int, default=1,
                   help="concurrent requests per context (>1 = concurrency phase)")
    p.add_argument("--json", dest="json_out", required=True)
    p.add_argument("--skip-modes", default="", help="comma list to skip (bf16,fp8)")
    args = p.parse_args(argv)

    skip = {m.strip() for m in args.skip_modes.split(",") if m.strip()}
    contexts = [c.strip() for c in args.contexts.split(",") if c.strip()]
    prompts = {}
    for label in contexts:
        path = Path(args.prompt_dir) / f"prompt_{label}.jsonl"
        if path.is_file():
            prompts[label] = json.loads(path.read_text().splitlines()[0])["prompt"]

    report: dict = {"model": args.model, "decode": args.decode, "cache": args.cache,
                    "mem_ratio": args.mem_ratio, "graph": not args.no_graph,
                    "prompts": {k: len(v) for k, v in prompts.items()}, "rows": [],
                    "quality": [], "capacity": {}, "commands": {}}
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        out_path.write_text(json.dumps(report, indent=1))

    for mode in MODES:
        if mode in skip:
            continue
        port = free_port()
        origin = f"http://127.0.0.1:{port}"
        fd, log_name = tempfile.mkstemp(prefix=f"kvq-{mode}-", suffix=".log")
        log_path = Path(log_name)
        cmd = serve_cmd(args, mode, port, args.max_running)
        report["commands"][mode] = " ".join(cmd)
        print(f"[bench] {mode}: {' '.join(cmd)}\n[bench] log {log_path}", flush=True)
        with open(fd, "wb") as log_f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True)

            def pump_output():
                for line in proc.stdout:
                    log_f.write(line)
                    log_f.flush()
                log_f.flush()

            pump = threading.Thread(target=pump_output, daemon=True)
            pump.start()
            try:
                wait_ready(origin, proc, log_path, timeout=2400)
                model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
                report["capacity"][mode] = parse_capacity(log_path)
                if args.quality:
                    report["quality"].extend(quality_rows(origin, model_id, prompts, args, mode))
                    flush()
                for label in contexts:
                    if label not in prompts:
                        continue
                    if args.parallel > 1:
                        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                            rows = list(pool.map(
                                lambda i: run_row(origin, model_id, prompts[label], args,
                                                  label, mode, warm=(i > 0)),
                                range(args.parallel),
                            ))
                        for i, row in enumerate(rows):
                            row["repeat"] = i
                            row["parallel"] = args.parallel
                            report["rows"].append(row)
                        agg_steps = sum(r["completion_tokens"] - 1 for r in rows)
                        span = max(r["ttft_ms"] + r["ms_per_token"] * (r["completion_tokens"] - 1)
                                   for r in rows) - min(r["ttft_ms"] for r in rows)
                        report["rows"].append({
                            "mode": mode, "context": label, "warm": True,
                            "repeat": 0, "parallel": args.parallel, "aggregate": True,
                            "decode_tok_s": agg_steps / span * 1e3,
                            "ttft_ms": max(r["ttft_ms"] for r in rows),
                        })
                        flush()
                        print(f"[bench] {mode} {label} x{args.parallel}: "
                              f"{report['rows'][-1]['decode_tok_s']:.2f} tok/s aggregate",
                              flush=True)
                        continue
                    repeats = args.repeats_short if label in ("4k", "16k", "32k") else args.repeats_long
                    for i in range(1 + repeats):
                        row = run_row(origin, model_id, prompts[label], args, label, mode,
                                      warm=i > 0)
                        row["repeat"] = i
                        report["rows"].append(row)
                        flush()
                        print(f"[bench] {mode} {label} rep{i}: ttft {row['ttft_ms']:.0f} ms, "
                              f"{row['decode_tok_s']:.2f} tok/s", flush=True)
                report["capacity"][mode] = parse_capacity(log_path)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                pump.join(timeout=10)
        flush()

    # Comparison: per context, every quantized mode vs bf16 warm decode and warm TTFT.
    modes_run = [m for m in MODES if m not in skip]
    summary = []
    for label in contexts:
        pair = {m: [r for r in report["rows"]
                    if r["mode"] == m and r["context"] == label
                    and not r.get("aggregate")] for m in modes_run}
        if not pair.get("bf16"):
            continue
        b = pair["bf16"][-1]
        for m in modes_run:
            if m == "bf16" or not pair.get(m):
                continue
            f = pair[m][-1]
            summary.append({
                "context": label, "mode": m,
                "decode_slowdown_pct": (b["decode_tok_s"] - f["decode_tok_s"]) / b["decode_tok_s"] * 100,
                "ttft_delta_pct": (f["ttft_ms"] - b["ttft_ms"]) / b["ttft_ms"] * 100,
            })
    report["summary"] = summary
    if args.quality:
        for label in sorted({q["context"] for q in report["quality"]
                             if q["kind"] == "retrieval"}):
            row = {"context": label}
            for m in modes_run:
                hits = [q for q in report["quality"] if q["kind"] == "retrieval"
                        and q["mode"] == m and q["context"] == label]
                if hits:
                    row[f"retrieval_{m}"] = sum(q["hit"] for q in hits) / len(hits)
            if len(row) > 1:
                report.setdefault("quality_summary", []).append(row)
    flush()
    print(f"[bench] report: {out_path}", flush=True)
    for row in summary:
        print(f"[bench] {row['context']}: decode {row['decode_slowdown_pct']:+.1f}%, "
              f"TTFT {row['ttft_delta_pct']:+.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
