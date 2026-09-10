"""Single-stream (bs=1) decode benchmark for any MoE model on any offload backend.

Measures through the real serving path: for each backend the bench spawns ``ft serve``,
sends a warmed chat request over /v1/chat/completions with ``stream=true``, and
timestamps every SSE event as it arrives. Numbers therefore include the scheduler,
detokenizer, and HTTP/SSE hop -- what a client actually sees -- not bare engine forwards.

Method -- at bs=1 the server emits one delta event per decode step, and the final chunk
(``stream_options.include_usage``) reports exact token counts, so

    decode_tok_s = (completion_tokens - 1) / (t_last_event - t_first_event)

which stays correct even when the detokenizer coalesces a few tokens into one event
(multibyte characters): the window is still anchored on the first and last token's
arrival. ``ignore_eos`` keeps the step count at exactly ``D`` regardless of sampling.
TTFT is the measured run's warm first-token latency (template rendering + prefill
included). Engine-internal diagnostics (expert-cache miss rate, hybrid fetch split) are
not exposed over the API and are not reported; VRAM is the server's live /v1/stats figure.

Prompt: an AIME-25 problem sent as a chat message with thinking enabled -- a real
reasoning workload, so expert routing is representative. The server renders the chat
template (including checkpoint-shipped encoders like DSV4's ``encoding_dsv4.py``). The
problems come from the ``math-ai/aime25`` dataset on the Hub, downloaded into the usual
HF cache on first run; ``--aime`` points at a local jsonl instead.

Sampling: the checkpoint's recommended params (``generation_config.json``), falling back
to temperature 1.0 / top_p 0.95 / top_k 64 for fields the checkpoint does not specify --
resolved here and sent explicitly, because the server's own unspecified-field defaults
are greedy and would silently degrade the routing workload for checkpoints without a
full sampling recommendation. The generated text is per-server-process deterministic
(fresh server, fixed request sequence), so one text sha1 per backend is a real
cross-backend check; token ids are not visible over the API, so this is a weaker
invariant than the old in-process id hash. ``--greedy`` sends temperature 0 for the
stricter comparison.

Run (one backend):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_decode_moe.py \
        --model /path/to/model

Run (all three backends, one server per backend):
    ... --model /path/to/model --backend offload,cpu,hybrid --json out.json

Speculative decoding (MTP): ``--spec-depth 0,2`` runs plain and depth-2 servers in one
invocation. Acceptance is parsed from each spawned server's ``spec accept: A/P`` log
line (API does not expose it), and every depth > 0 run is quality-gated against the
depth-0 output: a greedy spec run whose text shares less than ``--min-prefix-ratio``
of the plain output's characters as a common prefix fails the run (a broken verifier
collapses to ~0). Note that spec decode is not bit-identical to plain by design
(batched rows round differently), so the gate is a floor, not an equality check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Applied for every field the checkpoint's generation_config.json does not specify.
FALLBACK_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

# AIME-25 problems, pulled from the Hub into the usual HF cache on first run.
AIME_REPO = "math-ai/aime25"
AIME_FILE = "test.jsonl"
# Reasoning models need the answer format spelled out; the boxed answer is also what makes
# a run spot-checkable by eye.
BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="checkpoint dir (or .ftw)")
    p.add_argument(
        "--backend",
        default="offload",
        help="comma list of offload|cpu|hybrid; one server per backend",
    )
    p.add_argument(
        "--aime",
        default=os.environ.get("FREETOKEN_AIME25_JSONL"),
        help=f"local jsonl instead of downloading {AIME_REPO}; default $FREETOKEN_AIME25_JSONL",
    )
    p.add_argument("--problem", type=int, default=0, help="0-based AIME problem index")
    p.add_argument("--decode", type=int, default=256, help="decode tokens to measure (D)")
    p.add_argument(
        "--cache",
        type=int,
        default=0,
        help="GPU expert cache slots; 0 = auto-size from free VRAM",
    )
    p.add_argument("--cache-rate", type=float, default=None, help="cache slots as a fraction of L*E")
    p.add_argument(
        "--hybrid-fetch",
        type=int,
        default=-1,
        help="hybrid: max PCIe fetches/layer; -1 = auto (benched pcie/cpu bandwidth fraction)",
    )
    p.add_argument("--mem-ratio", type=float, default=0.9, help="target VRAM utilization")
    p.add_argument("--gpu", default=None,
                   help="GPU for the serve: a UUID or nvidia-smi index (as ft serve --gpu)")
    p.add_argument("--no-graph", action="store_true", help="eager decode instead of CUDA graph")
    p.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=["auto", "bf16", "fp8"],
        help="server --kv-cache-dtype (fp8 halves the KV bytes where the pool supports it)",
    )
    p.add_argument(
        "--dump-text", default=None,
        help="write the measured run's generated text here (quality comparisons)",
    )
    p.add_argument(
        "--max-extend-length", type=int, default=0,
        help="server --max-extend-length (prefill chunk token budget); 0 = server default (8192)",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="server --max-seq-len-override; 0 = 8192 + --decode (the decode-bench default)",
    )
    p.add_argument(
        "--spec-depth",
        default="0",
        help="comma list of MTP draft depths; each depth gets its own server run "
        "(0 = plain decode). Acceptance is parsed from the spawned server's log; a "
        "depth > 0 run is quality-gated against the depth-0 output.",
    )
    p.add_argument(
        "--spec-adaptive",
        action="store_true",
        help="add --speculative-adaptive to spec servers (depth tracks acceptance)",
    )
    p.add_argument(
        "--min-prefix-ratio",
        type=float,
        default=0.15,
        help="quality gate: fail a spec run whose greedy output shares less than this "
        "fraction of the depth-0 output's characters as a common prefix "
        "(a broken verifier collapses to ~0)",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="force temperature 0 (ignore the checkpoint's sampling) so ids are comparable",
    )
    p.add_argument(
        "--server-timeout",
        type=float,
        default=1800,
        help="seconds to wait for the spawned server to become ready",
    )
    p.add_argument("--json", dest="json_out", default=None, help="append the result rows here")
    return p.parse_args(argv)


def load_problem(path: str | None, index: int) -> tuple[str, str]:
    """One AIME-25 (problem, answer). Downloads the dataset unless ``path`` overrides it.

    Accepts both the Hub schema (``problem``) and the pre-formatted jsonl some local copies
    use (``prompt``, answer instruction already appended)."""
    if not path:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(AIME_REPO, AIME_FILE, repo_type="dataset")
        except Exception as e:  # offline, rate-limited, repo moved
            sys.exit(f"could not fetch {AIME_REPO}/{AIME_FILE} ({e}); pass --aime <local jsonl>")
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not 0 <= index < len(rows):
        sys.exit(f"--problem {index} out of range ({len(rows)} problems available)")
    row = rows[index]
    text = row.get("problem") or row["prompt"]
    if "boxed" not in text:
        text = f"{text}\n{BOXED_INSTRUCTION}"
    return text, str(row.get("answer", ""))


def resolve_sampling(model_path: str, greedy: bool) -> tuple[dict, str]:
    """Checkpoint-recommended sampling with per-field fallback; returns (params, source).

    Resolved client-side and sent explicitly: the server fills unspecified fields with
    its framework defaults (temperature 0 / no filtering), not with these fallbacks."""
    if greedy:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "greedy (--greedy)"
    recommended: dict = {}
    cfg = Path(model_path) / "generation_config.json"
    if cfg.is_file():
        raw = json.loads(cfg.read_text())
        recommended = {k: raw[k] for k in FALLBACK_SAMPLING if raw.get(k) is not None}
        if raw.get("do_sample") is False or recommended.get("temperature") == 0.0:
            return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "checkpoint (greedy)"
    params = {**FALLBACK_SAMPLING, **recommended}
    if params["top_k"] == 0:
        params["top_k"] = -1  # HF spells "no top-k filtering" as 0; the API as -1
    taken = sorted(recommended)
    source = f"checkpoint{taken} + fallback" if taken else "fallback (no generation_config)"
    return params, source


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve_cmd(args: argparse.Namespace, backend: str, port: int, depth: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "freetoken.cli", "serve",
        "--model", args.model,
        "--host", "127.0.0.1", "--port", str(port),
        "--moe-backend", backend,
        "--max-running-requests", "1",
        "--max-seq-len-override", str(args.max_seq_len or (8192 + args.decode)),
        "--memory-ratio", str(args.mem_ratio),
        "--cuda-graph-max-bs", "0" if args.no_graph else "1",
        "--moe-hybrid-max-fetch", str(args.hybrid_fetch),
    ]
    if args.kv_cache_dtype != "auto":
        cmd += ["--kv-cache-dtype", args.kv_cache_dtype]
    if args.max_extend_length > 0:
        cmd += ["--max-extend-length", str(args.max_extend_length)]
    if depth > 0:
        cmd += ["--mtp-depth", str(depth)]
        if args.spec_adaptive:
            cmd.append("--speculative-adaptive")
    if args.gpu:
        cmd += ["--gpu", args.gpu]
    if args.cache > 0:
        cmd += ["--moe-cache-size", str(args.cache)]
    elif args.cache_rate is not None:
        cmd += ["--moe-cache-rate", str(args.cache_rate)]
    else:
        cmd.append("--moe-cache-auto")
    return cmd


def parse_spec_acceptance(log_path: str) -> dict | None:
    """The last cumulative ``spec accept: A/P`` readout of a spawned spec server.

    The scheduler prints it on the periodic decode log line; counters are cumulative
    for the process, so the last line is the run's total. None for plain servers.
    """
    accepted = proposed = 0
    seen = False
    for line in Path(log_path).read_text(errors="ignore").splitlines():
        match = re.search(r"spec accept: (\d+)/(\d+)", line)
        if match:
            accepted, proposed, seen = int(match.group(1)), int(match.group(2)), True
    if not seen:
        return None
    return {
        "accepted": accepted,
        "proposed": proposed,
        "rate": accepted / proposed if proposed else 0.0,
    }


def quality_prefix(plain: str, spec: str) -> tuple[int, float]:
    """(common prefix chars, ratio of the plain output's length) for the quality gate."""
    n = 0
    while n < min(len(plain), len(spec)) and plain[n] == spec[n]:
        n += 1
    return n, (n / len(plain) if plain else 0.0)


def die_with_log(msg: str, log_path: str) -> None:
    tail = "".join(Path(log_path).read_text().splitlines(keepends=True)[-30:])
    sys.exit(f"[bench] {msg}\n[bench] server log tail ({log_path}):\n{tail}")


def wait_ready(origin: str, proc: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die_with_log(f"server exited with code {proc.returncode} during startup", log_path)
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except (OSError, ValueError):  # not bound yet / reset / partial response
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            die_with_log(f"server reported startup error: {health}", log_path)
        if health.get("maintenance") == "serving":
            return
        time.sleep(1.0)
    die_with_log(f"server not ready after {timeout:.0f}s", log_path)


def pump_output(src, log_f) -> None:
    """Mirror the server's output to our terminal while keeping the log file complete.

    Raw byte chunks (read1, not line-buffered) so \\r progress bars render live."""
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)
        log_f.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()


def stop_server(proc: subprocess.Popen) -> None:
    """SIGTERM the whole session (frontend + scheduler/tokenizer workers), escalate.

    Best-effort by design: it runs in ``finally`` and must not mask the real error.
    killpg runs even when the frontend already exited -- a crashed frontend leaves live
    non-daemon workers in the group, and they hold the GPU."""
    for sig, wait_s in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:  # whole group already gone
            pass
        try:
            proc.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    time.sleep(3)  # let the driver reclaim VRAM before the next backend's server


def stream_generate(origin: str, model_id: str, problem: str, sampling: dict,
                    args: argparse.Namespace) -> dict:
    """One streamed chat completion; returns per-token arrival stamps, text, and usage."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": args.decode,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling,
    }
    req = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict | None = None
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        sys.exit(f"[bench] request failed: HTTP {e.code}: {e.read()[:500]!r}")
    # Iterate the SSE stream line by line as bytes; json.loads decodes UTF-8 itself.
    # (A text-mode reader keyed off the content-type would decode latin-1: the server
    # sends ensure_ascii=False JSON with no charset on text/event-stream.)
    with resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue  # blank separators between events
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
    if usage is None:
        sys.exit("[bench] stream ended without a usage chunk; is this a FreeToken server?")
    return {"t0": t0, "stamps": stamps, "text": "".join(pieces), "usage": usage}


def run_one(args: argparse.Namespace, backend: str, depth: int) -> dict:
    problem, answer = load_problem(args.aime, args.problem)
    sampling, sampling_src = resolve_sampling(args.model, args.greedy)
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    fd, log_path = tempfile.mkstemp(prefix=f"bench-serve-{backend}-", suffix=".log")
    cmd = serve_cmd(args, backend, port, depth)

    print(
        f"[bench] model={args.model}\n"
        f"[bench] backend={backend} depth={depth} "
        f"cache={args.cache or args.cache_rate or 'auto'} "
        f"mem_ratio={args.mem_ratio} decode={args.decode} graph={not args.no_graph}\n"
        f"[bench] sampling={sampling} <- {sampling_src}\n"
        f"[bench] server log: {log_path}",
        flush=True,
    )

    with os.fdopen(fd, "wb") as log_f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
        )
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        try:
            wait_ready(origin, proc, log_path, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            print(f"[bench] model_id={model_id}", flush=True)
            print(f"[bench] AIME25 #{args.problem} (answer {answer})", flush=True)

            # Warm the expert cache to a steady-state decode working set. The warmup's
            # first token is the COLD TTFT: empty radix prefix, first prefill after load.
            warmup = stream_generate(origin, model_id, problem, sampling, args)
            r = stream_generate(origin, model_id, problem, sampling, args)
            stats = get_json(f"{origin}/v1/stats")
            acc = parse_spec_acceptance(log_path) if depth > 0 else None
        finally:
            stop_server(proc)
            pump.join(timeout=10)

    stamps, usage = r["stamps"], r["usage"]
    if len(stamps) < 2:
        sys.exit(f"[bench] need >=2 token events to measure decode, got {len(stamps)}")
    completion = usage["completion_tokens"]
    if completion != args.decode:
        print(f"[bench] WARNING: completion_tokens={completion} != --decode {args.decode}", flush=True)
    steps = completion - 1
    decode_time = stamps[-1] - stamps[0]
    gaps = sorted((b - a) * 1e3 for a, b in zip(stamps, stamps[1:]))
    row = {
        "model": args.model,
        "backend": backend,
        "mtp_depth": depth,
        "spec_accept": acc,
        "spec_accept_rate": None if acc is None else round(acc["rate"], 4),
        "problem": args.problem,
        "prompt_tokens": usage["prompt_tokens"],
        "decode_steps": steps,
        "decode_tok_s": steps / decode_time if decode_time > 0 else 0.0,
        "ms_per_token": decode_time / steps * 1e3 if steps > 0 else 0.0,
        "event_ms_p50": gaps[len(gaps) // 2],
        "event_ms_p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))],
        "ttft_ms": (stamps[0] - r["t0"]) * 1e3,
        "ttft_cold_ms": (
            (warmup["stamps"][0] - warmup["t0"]) * 1e3 if warmup["stamps"] else None
        ),
        "events": len(stamps),
        "completion_tokens": completion,
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "sampling": sampling,
        "output_sha1": hashlib.sha1(r["text"].encode()).hexdigest()[:12],
        "server_log": log_path,
    }

    print(f"\n==== decode bs=1 [{backend}] depth={depth} via /v1/chat/completions ====", flush=True)
    print(f"  decode throughput : {row['decode_tok_s']:8.2f} tok/s  ({row['ms_per_token']:.3f} ms/token)")
    if acc is not None:
        print(f"  spec acceptance   : {acc['rate']:8.3f}  ({acc['accepted']}/{acc['proposed']})")
    cold_s = f"{row['ttft_cold_ms']:.1f}" if row["ttft_cold_ms"] is not None else "n/a"
    print(f"  TTFT cold/warm    : {cold_s}/{row['ttft_ms']:.1f} ms  (prompt {row['prompt_tokens']} tok)")
    print(f"  decode measured   : {steps} steps in {decode_time:.3f} s  "
          f"(event p50 {row['event_ms_p50']:.3f} / p99 {row['event_ms_p99']:.3f} ms, "
          f"{len(stamps)} events)")
    print(f"  vram (server)     : {row['vram_gib']:8.2f} GiB")
    if args.dump_text:
        Path(args.dump_text).write_text(r["text"])
    sha_note = "greedy" if args.greedy else "sampled, per-server deterministic"
    print(f"  output sha1       : {row['output_sha1']}  ({sha_note}; compare across backends)")
    print(f"  output sample     : {r['text'][:240]!r}")
    return {**row, "_text": r["text"]}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backends = [b.strip() for b in args.backend.split(",") if b.strip()]
    unknown = [b for b in backends if b not in ("offload", "cpu", "hybrid")]
    if unknown:
        sys.exit(f"unknown backend(s): {unknown}")
    depths = [int(d) for d in str(args.spec_depth).split(",") if d.strip()]
    if any(d < 0 for d in depths):
        sys.exit("--spec-depth must be non-negative")

    failed = []
    plain_text: dict[str, str] = {}
    for backend in backends:
        for depth in depths:
            try:
                row = run_one(args, backend, depth)
            # SystemExit inherits BaseException, not Exception, so name both: a mid-decode
            # connection drop (server crash) must not abort the remaining runs either.
            except (SystemExit, Exception) as e:
                if len(backends) == 1 and len(depths) == 1:
                    raise
                print(f"\n[bench] {backend} depth={depth} failed: {e!r}", flush=True)
                failed.append(f"{backend} d{depth}")
                continue
            text = row.pop("_text")
            if depth == 0:
                plain_text[backend] = text
            elif backend in plain_text:
                chars, ratio = quality_prefix(plain_text[backend], text)
                row["quality_prefix_chars"] = chars
                row["quality_prefix_ratio"] = round(ratio, 4)
                print(
                    f"  quality vs plain  : {chars} chars common prefix "
                    f"({ratio:.2f} of the plain output)",
                    flush=True,
                )
                if ratio < args.min_prefix_ratio:
                    print(
                        f"\n[bench] QUALITY GATE: depth={depth} prefix ratio {ratio:.2f} "
                        f"< --min-prefix-ratio {args.min_prefix_ratio}: the verifier looks "
                        f"broken (acceptance {row.get('spec_accept_rate')})",
                        flush=True,
                    )
                    failed.append(f"{backend} d{depth} (quality)")
            if args.json_out:
                with open(args.json_out, "a") as f:
                    f.write(json.dumps(row) + "\n")
    if failed:
        print(f"\n[bench] runs that failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
