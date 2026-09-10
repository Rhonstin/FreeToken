from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Set

from freetoken.core import Batch, Req
from freetoken.engine.mtp import clamp_spec_depth


@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)
    # Optional speculative (MTP) depth provider: returns the draft-row count for one
    # request, or None / 0 to keep it a plain decode step. Installed by the engine when
    # a draft model is attached. Plain (None) behavior is byte-identical to the old path.
    spec_depth_fn: Callable[[Req], int] | None = None

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        # A speculative step's reserved draft rows (device_len - cached_len - 1 per request;
        # 0 for plain decode, whose post-step frame is exactly cached_len + 1) hold allocated
        # pages until the verify commit releases the rejected tail, so admission counts them.
        spec_reserved = sum(req.device_len - req.cached_len - 1 for req in self.running_reqs)
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved + spec_reserved

    @staticmethod
    def _spec_rows(reqs: list[Req]) -> list[tuple[Req, int]]:
        rows: list[tuple[Req, int]] = []
        for req in reqs:
            for spec_index in range(req.spec_depth + 1):
                rows.append((req, spec_index))
        return rows

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        reqs = sorted(self.running_reqs, key=lambda req: req.uid)
        spec_rows: list[tuple[Req, int]] | None = None
        if self.spec_depth_fn is not None:
            # Re-establish every step's frame from cached_len (see Req.reserve_spec): the
            # anchor row keeps a plain decode's contract, each draft row extends it one
            # position further. Plain decode passes through unchanged since the
            # post-commit frame already equals cached_len + 1.
            for req in reqs:
                depth = clamp_spec_depth(self.spec_depth_fn(req), req.cached_len, req.max_device_len)
                req.reserve_spec(depth)
            if any(req.spec_depth for req in reqs):
                spec_rows = self._spec_rows(reqs)
        return Batch(reqs=reqs, phase="decode", spec_rows=spec_rows)

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
