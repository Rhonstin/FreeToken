"""Draft-free speculation: prompt look-up (PLD) proposals for the multi-token verify frame.

The drafter keeps, per request, the committed token sequence and a map from each
n-gram (up to ``max_ngram`` tokens) to its most recent occurrence. A step proposes the
continuation of the most recent EARLIER occurrence of the request's longest suffix
(shortest accepted match ``min_ngram``), up to the configured depth.

Proposals cost no VRAM and no model forward. Verification is the same exact path MTP
uses, so output quality is unchanged: a rejected draft only costs the verify forward's
rows (expert fetches on the offload box), which is why weak matches are excluded by
``min_ngram`` -- on non-repetitive prose no match means no draft and the step stays a
plain decode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class _ReqIndex:
    """One request's committed tokens plus the n-gram -> last-ending-position map."""

    tokens: list[int] = field(default_factory=list)
    # (k-gram tuple) -> (latest end, previous end). Two slots: the suffix's own
    # occurrence (at the tip) must not hide the earlier match a proposal needs.
    positions: dict[tuple[int, ...], tuple[int, int | None]] = field(default_factory=dict)
    indexed: int = 0  # tokens[:indexed] are in ``positions``

    def sync(self, ids: torch.Tensor, max_ngram: int) -> None:
        """Extend the index with tokens committed since the last call (or rebuild on a
        rewind, e.g. a test reusing a uid). Indexes every token: the suffix that
        matters ends at the caller-supplied anchor tip, not at the indexed tip."""
        n = int(ids.numel())
        if n < self.indexed:
            self.tokens = self.tokens[:n]
            self.positions.clear()
            self.indexed = 0
        if n == self.indexed:
            return
        fresh = ids[self.indexed : n].tolist()
        start = self.indexed
        self.tokens.extend(fresh)
        for offset in range(len(fresh)):
            end = start + offset
            for k in range(1, min(max_ngram, end + 1) + 1):
                key = tuple(self.tokens[end - k + 1 : end + 1])
                prev = self.positions.get(key)
                self.positions[key] = (end, prev[0] if prev is not None else None)
        self.indexed = n

    def propose(self, depth: int, max_ngram: int, min_ngram: int, tip: int) -> list[int]:
        """Up to ``depth`` tokens following the best earlier occurrence of the suffix
        ending at ``tip`` (longest suffix first). Empty when no earlier occurrence of
        at least ``min_ngram`` tokens exists.

        ``tip`` is the anchor token being continued. It usually IS the last indexed
        token, but under overlap scheduling a plain step's sampled anchor is not yet in
        ``input_ids`` at plan time -- the suffix is then built from the idle history
        plus the tip, keeping the proposals aligned with the positions they predict.
        """
        n = len(self.tokens)
        base = n - 1 if n and self.tokens[-1] == tip else n
        if depth <= 0 or base < min_ngram:
            return []
        for k in range(min(max_ngram, base), min_ngram - 1, -1):
            suffix = tuple(self.tokens[base - k + 1 : base]) + (tip,)
            ends = self.positions.get(suffix)
            if ends is None:
                continue
            # The occurrence at or past ``base - k`` is the current suffix itself.
            end = max((e for e in ends if e is not None and e <= base - k), default=None)
            if end is None:
                continue
            got = self.tokens[end + 1 : end + 1 + depth]
            if got:
                return got
        return []


class LookupDrafter:
    """Per-request prompt look-up drafter; the scheduler owns one instance per run."""

    def __init__(self, max_ngram: int = 6, min_ngram: int = 2):
        assert 1 <= min_ngram <= max_ngram, (min_ngram, max_ngram)
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self._index: dict[int, _ReqIndex] = {}
        self._planned: dict[int, list[int]] = {}

    def plan(self, req, depth: int, anchor_token: int) -> int:
        """Schedule-time hook (runs as ``DecodeManager.spec_depth_fn``): sync the
        index from ``req.input_ids``, stash up to ``depth`` draft tokens continuing
        ``anchor_token``, and return how many rows this step will actually carry
        (0 keeps the request a plain row)."""
        if depth <= 0:
            return 0
        state = self._index.setdefault(req.uid, _ReqIndex())
        state.sync(req.input_ids, self.max_ngram)
        drafts = state.propose(depth, self.max_ngram, self.min_ngram, anchor_token)
        if not drafts:
            return 0
        self._planned[req.uid] = drafts
        return len(drafts)

    def take(self, uid: int, depth: int) -> list[int]:
        """The stashed drafts for this step, capped at the (clamped) scheduled depth."""
        return self._planned.pop(uid, [])[:depth]

    def drop(self, uid: int) -> None:
        self._index.pop(uid, None)
        self._planned.pop(uid, None)
