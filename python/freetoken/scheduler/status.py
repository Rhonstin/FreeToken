from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from freetoken.core import Batch


@dataclass
class SchedulerStatusReporter:
    log: Callable[[str], None]
    clock: Callable[[], float] = time.perf_counter
    decode_log_interval: int = 40
    _last_prefill_time: float = field(init=False)
    _last_decode_time: float = field(init=False)
    _decode_forward_count: int = field(default=0, init=False)
    _decode_generated_tokens: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._last_prefill_time = now
        self._last_decode_time = now
        self.decode_log_interval = max(1, self.decode_log_interval)

    def report_batch(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        spec: dict | None = None,
        moe: dict | None = None,
    ) -> None:
        if batch.is_prefill:
            self._report_prefill(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
            )
        elif batch.is_decode:
            self._report_decode(
                batch,
                running_reqs=running_reqs,
                queue_reqs=queue_reqs,
                kv_used_pages=kv_used_pages,
                kv_total_pages=kv_total_pages,
                page_size=page_size,
                mamba_slots=mamba_slots,
                swa_tokens=swa_tokens,
                spec=spec,
                moe=moe,
            )

    def _report_prefill(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
    ) -> None:
        now = self.clock()
        gap = now - self._last_prefill_time
        self._last_prefill_time = now
        # Read the schedule-time snapshot: by report time the forward's complete_one() has
        # advanced each req's cached_len to device_len, so reading the reqs here would log
        # decode-state values (#new-token == #reqs, #cached-token == full prompt).
        new_tokens = batch.log_new_tokens
        cached_tokens = batch.log_cached_tokens
        input_throughput = new_tokens / gap if gap > 0 else 0.0
        self.log(
            f"Prefill batch, "
            f"#new-seq: {len(batch.reqs)}, "
            f"#new-token: {new_tokens}, "
            f"#cached-token: {cached_tokens}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"#running-req: {running_reqs}, "
            f"#queue-req: {queue_reqs}, "
            f"input throughput (token/s): {input_throughput:.2f}"
        )

    def _report_decode(
        self,
        batch: Batch,
        *,
        running_reqs: int,
        queue_reqs: int,
        kv_used_pages: int,
        kv_total_pages: int,
        page_size: int,
        mamba_slots: tuple[int, int] | None = None,
        swa_tokens: tuple[int, int] | None = None,
        spec: dict | None = None,
        moe: dict | None = None,
    ) -> None:
        self._decode_forward_count += 1
        # A finalized spec step generates its whole accepted span, not one token per
        # request: count the accepted tokens beyond the anchor so gen throughput is real
        # tok/s under speculation. Plain batches (no spec_accepted) are unchanged.
        self._decode_generated_tokens += len(batch.reqs) + _spec_extra_tokens(batch)
        if self._decode_forward_count % self.decode_log_interval != 0:
            return

        now = self.clock()
        gap = now - self._last_decode_time
        self._last_decode_time = now
        gen_throughput = self._decode_generated_tokens / gap if gap > 0 else 0.0
        self._decode_generated_tokens = 0
        self.log(
            f"Decode batch, "
            f"#running-req: {running_reqs}, "
            f"#token: {kv_used_pages * page_size}, "
            f"token usage: {_usage_ratio(kv_used_pages, kv_total_pages):.2f}, "
            f"{_swa_msg(swa_tokens)}"
            f"{_mamba_msg(mamba_slots)}"
            f"gen throughput (token/s): {gen_throughput:.2f}, "
            f"{_spec_msg(spec)}"
            f"{_moe_msg(moe)}"
            f"#queue-req: {queue_reqs}"
        )


def _usage_ratio(used: int, total: int) -> float:
    return used / total if total > 0 else 0.0


def _spec_extra_tokens(batch: Batch) -> int:
    """Accepted tokens beyond one per request on a finalized spec batch, else 0.

    Reads the verify driver's ``spec_accepted`` map (uid -> accepted span, which always
    includes the trailing resample/bonus token); plain batches carry no map and count
    exactly as before. Duck-typed via getattr so schedule-time test doubles without
    the field keep working.
    """
    accepted = getattr(batch, "spec_accepted", None)
    if not accepted:
        return 0
    return sum(len(span) - 1 for span in accepted.values())


def _spec_msg(spec: dict | None) -> str:
    """The MTP acceptance-rate fragment for the decode log line (the 419.8 benchmark
    reads the same ``SpecAccounting.snapshot`` shape); empty unless steps verified, so
    plain-decode lines are byte-identical."""
    if not spec or not spec.get("steps"):
        return ""
    return (
        f"spec accept: {spec['accepted']}/{spec['proposed']} "
        f"({spec['accepted'] / spec['proposed'] if spec['proposed'] else 0.0:.2f}), "
    )


def _moe_msg(moe: dict | None) -> str:
    """Opt-in MoE cache readout (--moe-collect-stats): realized miss rate, the routing
    working set / 90%-mass expert count, and the per-layer-LRU oracle hit at the
    current slots per layer. Empty unless the flag is on, so plain lines stay
    byte-identical."""
    if not moe:
        return ""
    parts = []
    if "miss_rate" in moe:
        parts.append(f"miss: {moe['miss_rate']:.2f}")
    if "oracle_hit_at_slots" in moe:
        parts.append(
            f"oracle: {moe['oracle_hit_at_slots']:.2f}@{moe['slots_per_layer']:.0f}slots")
    if "working_set_mean" in moe:
        parts.append(f"ws: {moe['working_set_mean']:.0f}/{moe['working_set_max']}")
    if "experts_for_90pct" in moe:
        parts.append(f"e90: {moe['experts_for_90pct']:.0f}")
    if "norm_entropy" in moe:
        parts.append(f"ent: {moe['norm_entropy']:.2f}")
    return "moe " + ", ".join(parts) + ", " if parts else ""


def _mamba_msg(mamba_slots: tuple[int, int] | None) -> str:
    """GDN-state (mamba) pool occupancy for hybrid models; empty for the rest."""
    if mamba_slots is None:
        return ""
    used, total = mamba_slots
    return f"#mamba-slot: {used}/{total}, mamba usage: {_usage_ratio(used, total):.2f}, "


def _swa_msg(swa_tokens: tuple[int, int] | None) -> str:
    """Window (swa) pool occupancy for SWA models; empty for the rest."""
    if swa_tokens is None:
        return ""
    used, total = swa_tokens
    return f"#swa-token: {used}/{total}, swa usage: {_usage_ratio(used, total):.2f}, "
