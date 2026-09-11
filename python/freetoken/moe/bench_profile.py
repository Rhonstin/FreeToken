"""Torch-free reader for the ``ft bench bw`` hardware profile (``benchbw/<gpu-uuid>.json``).

The engine consults this at MoE-backend *auto* resolution (``engine.py``) to make the
offload-vs-hybrid choice hardware-adaptive without importing the (torch-heavy) benchmark
itself. ``benchbw.py`` writes the profile; this module only reads it.

The join key is the expert *format*: the CPU-MoE-vs-PCIe-gather bandwidth ratio the choice
rides on is dominated by ``(format, hardware)``, not by the exact model, so a profile benched
on one workload transfers to any model with the same expert format on the same GPU.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess

from freetoken.utils import init_logger

logger = init_logger(__name__)

# Engine ``expert_quant`` (models/config.py) -> benchbw format key (offload_cache._BANK_SCHEMAS
# / benchbw._offload_bank_specs). Only the offload-family formats with a CPU MoE weight path can
# ever resolve to hybrid; anything not listed falls through unmapped and finds no profile entry
# (-> None -> offload), which is the safe default.
_QUANT_TO_BENCH_FORMAT = {
    "nvfp4": "nvfp4",
    "ds_fp4": "ds_fp4",
    "mxfp4": "mxfp4_triton",
    "bf16": "bf16",
    "fp8_block": "fp8_block",
}


def _cache_dir() -> str:
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "freetoken")


def default_profile_path(gpu_uuid: str | None = None) -> str:
    """``$XDG_CACHE_HOME/freetoken/benchbw/<gpu-uuid>.json``, or the legacy ``benchbw.json`` without a uuid.

    One file per GPU: bandwidth differs between slots.
    """
    if gpu_uuid:
        return os.path.join(_cache_dir(), "benchbw", f"{gpu_uuid}.json")
    return os.path.join(_cache_dir(), "benchbw.json")


def latest_profile_path() -> str | None:
    """Newest ``benchbw/*.json``, else the legacy ``benchbw.json``, else None."""
    per_gpu = os.path.join(_cache_dir(), "benchbw")
    newest: tuple[float, str] | None = None
    try:
        for name in os.listdir(per_gpu):
            if not name.endswith(".json"):
                continue
            path = os.path.join(per_gpu, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if newest is None or mtime > newest[0]:
                newest = (mtime, path)
    except OSError:
        pass
    if newest is not None:
        return newest[1]
    legacy = default_profile_path()
    return legacy if os.path.isfile(legacy) else None


def _load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ======================================================================================
# Hardware fingerprint (versioned) + pure compatibility matcher
# ======================================================================================
#
# A profile benched on one machine must not be reused on a different one just because
# the GPU name matches: the offload-vs-hybrid choice rides on CPU, RAM and PCIe too.
# The reader is torch-free; everything here reads /proc, sysfs, dmidecode and nvidia-smi
# and degrades to null (never a guess) when a source is unavailable.

_FP_SCHEMA = 5


def _cpuinfo() -> dict:
    """First processor's /proc/cpuinfo fields, or {} when unavailable."""
    try:
        with open("/proc/cpuinfo") as f:
            block = f.read().split("\n\n", 1)[0]
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _isa_tier(flags: str | None) -> str | None:
    """Highest CPU-MoE ISA tier the CPU advertises (matches benchbw._ISA_TIERS)."""
    if not flags:
        return None
    have = set(flags.split())
    if "avx512bf16" in have:
        return "avx512bf16"
    if "avx512f" in have:
        return "avx512"
    if "avx2" in have:
        return "avx2"
    return "scalar"


def _physical_cores() -> int | None:
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:
        allowed = list(range(os.cpu_count() or 1))
    seen: set[str] = set()
    for cpu in allowed:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                seen.add(f.read().strip())
        except OSError:
            seen.add(str(cpu))
    return len(seen) or None


def _mem_total_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _dmi_memory() -> tuple[int | None, int | None]:
    """(configured_speed_mts, observed_channels) from dmidecode when permitted, else (None, None).

    ``dmidecode`` needs root; try it directly, then ``sudo -n`` (never prompts). Both are
    best-effort: a permission failure yields nulls, not a guess.
    """
    for argv in (["dmidecode", "-t", "memory"], ["sudo", "-n", "dmidecode", "-t", "memory"]):
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if p.returncode != 0 or "Memory Device" not in p.stdout:
            continue
        speeds: set[int] = set()
        channels: set[str] = set()
        for block in p.stdout.split("Memory Device")[1:]:
            if "No Module" in block:
                continue
            m = re.search(r"\n\s+Configured Memory Speed:\s*(\d+)\s*MT/s", block)
            if m:
                speeds.add(int(m.group(1)))
            m = re.search(r"\n\s+Locator:\s*(\S+)", block)
            if m:
                channels.add(re.sub(r"\d+$", "", m.group(1).split("_")[-1]))
        speed = next(iter(speeds)) if len(speeds) == 1 else (max(speeds) if speeds else None)
        return speed, (len(channels) or None)
    return None, None


def _pcie_link(gpu_index: int = 0) -> dict:
    """Negotiated max PCIe link from nvidia-smi (nulls when unavailable)."""
    nulls = {"bdf": None, "gen_max": None, "width_max": None}
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id,pcie.link.gen.max,pcie.link.width.max",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if p.returncode != 0:
            return nulls
        bus, gen, width = (x.strip() for x in p.stdout.strip().splitlines()[gpu_index].split(","))
        return {"bdf": bus, "gen_max": int(gen), "width_max": int(width)}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return nulls


def read_machine_fingerprint(*, gpu_name: str | None = None, gpu_sm: str | None = None,
                             runtime: dict | None = None, expert: dict | None = None,
                             threads: int | None = None, concurrency: int = 1) -> dict:
    """The current machine's fingerprint (no hostname, serials or MAC).

    ``gpu_sm``/``runtime``/``expert`` come from the caller (they need torch/the workload);
    cpu/ram/pcie are read here. ``threads`` defaults to the physical-core count (the auto
    MoE thread default), so a profile benched at a different thread count will not match.
    """
    cpu = _cpuinfo()
    speed, channels = _dmi_memory()
    return {
        "schema_version": _FP_SCHEMA,
        "cpu": {
            "vendor": cpu.get("vendor_id"),
            "family": cpu.get("cpu family"),
            "model": cpu.get("model"),
            "stepping": cpu.get("stepping"),
            "sku": cpu.get("model name"),
            "isa": _isa_tier(cpu.get("flags")),
            "threads": threads if threads is not None else _physical_cores(),
        },
        "ram": {"total_bytes": _mem_total_bytes(), "observed_channels": channels,
                "configured_speed_mts": speed},
        "pcie": _pcie_link(),
        "gpu": {"name": gpu_name, "sm": gpu_sm},
        "runtime": runtime or {},
        "expert": expert or {},
        "concurrency": concurrency,
    }


_REQUIRED_FP = (("cpu", "family"), ("cpu", "model"), ("cpu", "stepping"), ("cpu", "isa"),
                ("cpu", "threads"), ("ram", "total_bytes"), ("gpu", "name"))
# Compared only when both sides recorded them (privilege-limited or caller-supplied).
_OPTIONAL_FP = (("cpu", "sku"), ("ram", "configured_speed_mts"), ("ram", "observed_channels"),
                ("pcie", "bdf"), ("pcie", "gen_max"), ("pcie", "width_max"), ("gpu", "sm"))


def profile_compatible(stored: dict, current: dict) -> tuple[bool, str]:
    """Pure fingerprint matcher: ``(ok, reason)``. No torch, no GPU.

    A required field missing on either side is unverified -> reject (never assumed equal).
    Optional fields mismatching when both are present -> reject. Every failure names the field.
    """
    if not isinstance(stored, dict) or stored.get("schema_version") != _FP_SCHEMA:
        return False, f"unknown/missing profile schema_version (want {_FP_SCHEMA})"
    if not isinstance(current, dict):
        return False, "no current machine fingerprint"
    for sec, key in _REQUIRED_FP:
        sv = (stored.get(sec) or {}).get(key)
        cv = (current.get(sec) or {}).get(key)
        if sv is None:
            return False, f"stored fingerprint missing {sec}.{key} (unverified)"
        if cv is None:
            return False, f"current machine cannot read {sec}.{key}"
        if sv != cv:
            return False, f"{sec}.{key} mismatch: stored={sv!r} current={cv!r}"
    for sec, key in _OPTIONAL_FP:
        sv = (stored.get(sec) or {}).get(key)
        cv = (current.get(sec) or {}).get(key)
        if sv is not None and cv is not None and sv != cv:
            return False, f"{sec}.{key} mismatch: stored={sv!r} current={cv!r}"
    return True, "compatible"


def profile_bandwidths_valid(prof: dict) -> tuple[bool, str]:
    """Finite, positive bandwidths; NaN/Inf/<=0 rejected."""
    def ok(x) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(x) and x > 0
    entries = list((prof.get("dtype_kernels") or {}).values())
    entries += [k for wl in (prof.get("workloads") or {}).values()
                if isinstance(wl, dict) for k in (wl.get("kernels") or {}).values()]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in ("cpu_moe_gbs", "pcie_gather_gbs"):
            v = entry.get(key)
            if v is not None and not ok(v):
                return False, f"{key}={v!r} is not finite and positive"
    return True, "ok"


def _usable_profile(
    gpu_name: str | None, path: str | None, gpu_uuid: str | None = None,
    current_fingerprint: dict | None = None,
) -> dict | None:
    """The cached profile, or ``None`` when there is no file / it is not valid for this machine.

    Validity: schema_version matches, the GPU name matches, the fingerprint is compatible
    with the current machine (CPU/RAM/PCIe/threads), and the bandwidths are finite. A legacy
    profile (no schema_version) is informational only and is ignored here. Every rejection
    logs the exact reason; the caller keeps the safe offload default.

    Lookup: explicit ``path`` (else ``FREETOKEN_BENCHBW_PATH``) -> ``benchbw/<gpu_uuid>.json`` -> legacy ``benchbw.json``.
    """
    explicit = path or os.environ.get("FREETOKEN_BENCHBW_PATH")
    if explicit:
        candidates = [explicit]
    else:
        candidates = [default_profile_path(gpu_uuid)] if gpu_uuid else []
        candidates.append(default_profile_path())
    prof = None
    for src in candidates:
        prof = _load(src)
        if isinstance(prof, dict):
            break
        if os.path.exists(src):
            # unreadable profile for this card: stay on the safe default, do not borrow the legacy file
            return None
    if not isinstance(prof, dict):
        return None
    prof_gpu = (prof.get("gpu") or {}).get("name")
    if gpu_name and prof_gpu and prof_gpu != gpu_name:
        logger.warning(
            f"benchbw profile {src} was measured on {prof_gpu!r}, not this GPU "
            f"({gpu_name!r}); ignoring it"
        )
        return None
    if prof.get("schema_version") != _FP_SCHEMA:
        logger.warning(
            f"benchbw profile {src} has no usable schema_version "
            f"(got {prof.get('schema_version')!r}, want {_FP_SCHEMA}); treating it as "
            f"informational only"
        )
        return None
    current = current_fingerprint if current_fingerprint is not None else read_machine_fingerprint(gpu_name=gpu_name)
    ok, reason = profile_compatible(prof.get("fingerprint") or {}, current)
    if not ok:
        logger.warning(f"benchbw profile {src} rejected: {reason}")
        return None
    ok, reason = profile_bandwidths_valid(prof)
    if not ok:
        logger.warning(f"benchbw profile {src} rejected: {reason}")
        return None
    return prof


def load_backend_recommendation(
    quant_format: str,
    gpu_name: str | None = None,
    path: str | None = None,
    gpu_uuid: str | None = None,
    current_fingerprint: dict | None = None,
) -> str | None:
    """Bench-recommended offload-family backend for ``quant_format`` on this GPU, or ``None``.

    Returns ``"hybrid"`` only when *every* benched workload sharing this expert format
    recommended hybrid (CPU MoE BW > threshold x PCIe gather BW); a mixed verdict (a
    near-threshold format) resolves conservatively to ``"offload"``. ``None`` means "no usable
    profile" (see ``_usable_profile``) or no entry for this format. The caller keeps its own
    default (offload) on ``None``.
    """
    fmt = _QUANT_TO_BENCH_FORMAT.get(quant_format, quant_format)
    prof = _usable_profile(gpu_name, path, gpu_uuid, current_fingerprint)
    if prof is None:
        return None

    # Preferred: the per-dtype tuning verdicts (`ft bench bw --dtype`), a direct format->backend
    # map -- the axis the backend pick is meant to key on.
    dtypes = prof.get("dtypes")
    if isinstance(dtypes, dict) and dtypes.get(fmt) in ("hybrid", "offload"):
        return dtypes[fmt]

    # Fallback: a per-model profile (`ft bench bw --model`). Aggregate the workloads sharing this
    # format -- unanimous hybrid -> hybrid; any offload (a near-threshold split) -> offload.
    workloads = prof.get("workloads")
    if not isinstance(workloads, dict):
        return None
    picks = [
        entry["recommended"]
        for wl in workloads.values()
        if isinstance(wl, dict)
        for entry in [(wl.get("kernels") or {}).get(fmt)]
        if isinstance(entry, dict) and entry.get("recommended")
    ]
    if not picks:
        return None
    return "hybrid" if all(p == "hybrid" for p in picks) else "offload"


def load_hybrid_fetch_fraction(
    quant_format: str,
    gpu_name: str | None = None,
    path: str | None = None,
    gpu_uuid: str | None = None,
    current_fingerprint: dict | None = None,
) -> float | None:
    """Benched hybrid fetch fraction for ``quant_format``, or ``None``.

    The hybrid backend's bandwidth-matched fetch split: of a decode step's expert misses,
    fetch this fraction over PCIe and compute the rest on the CPU, so both finish together.
    Preferred source is the *overlapped* pair (CPU MoE and PCIe gather measured while
    running concurrently -- the real contention regime): fetched/misses = pcie_ov /
    (pcie_ov + cpu_ov). Older profiles without it fall back to the standalone bandwidths
    under a full-DRAM-contention assumption (cpu keeps cpu - pcie under DMA), which
    reduces to pcie/cpu. Per-dtype entry first, then any per-model entry with this format.
    ``None`` = no usable profile; clamped to [0, 1].
    """
    fmt = _QUANT_TO_BENCH_FORMAT.get(quant_format, quant_format)
    prof = _usable_profile(gpu_name, path, gpu_uuid, current_fingerprint)
    if prof is None:
        return None
    entries = [(prof.get("dtype_kernels") or {}).get(fmt)] + [
        (wl.get("kernels") or {}).get(fmt)
        for wl in (prof.get("workloads") or {}).values()
        if isinstance(wl, dict)
    ]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        cpu_ov, pcie_ov = entry.get("cpu_moe_overlap_gbs"), entry.get("pcie_gather_overlap_gbs")
        if cpu_ov and pcie_ov:
            return min(1.0, pcie_ov / (pcie_ov + cpu_ov))
        cpu, pcie = entry.get("cpu_moe_gbs"), entry.get("pcie_gather_gbs")
        if cpu and pcie:
            return min(1.0, pcie / cpu)
    return None
