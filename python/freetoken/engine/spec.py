"""Probability-correct speculative verification and sampling.

The position-level core of MTP speculative decoding (Leviathan et al. / Chen et al.):
draft tokens proposed for the next ``d`` positions are accepted or rejected against the
target model's distributions, preserving the target distribution EXACTLY. A wrong rule
here does not crash -- it silently degrades quality -- so the contract below is pinned
by a distribution-equivalence test, not just by shape tests.

Indexing (positions, not rows): for a step opened at ``base_len`` with ``d`` drafts,
``draft_tokens[i]`` / ``draft_logits[i]`` propose the token for position
``base_len + 1 + i`` with draft distribution ``q_{i+1}``, and ``target_logits[i]`` is the
target distribution ``p_{i+1}`` for the same position; ``target_logits[d]`` is the bonus
distribution. How rows map to these tensors is the verify driver's business (for a
decode-rows forward, ``p_{i+1}`` is row ``i``'s logits, since row ``i`` consumes position
``base_len + i`` and predicts one past it); the verifier only sees positions.

Per position, with ``r ~ U(0, 1)``: accept ``x`` when ``r < min(1, p(x)/q(x))``; on
rejection resample from ``norm(max(0, p - q))`` and stop (later drafts are discarded);
when every draft is accepted, draw one bonus token from the last target row. The
accepted tokens always number ``n_accepted + 1`` (the resample/bonus included), so at
least one new token is produced per step and ``committed = base_len + len(accepted)``.

Both ``p`` and ``q`` are formed AFTER the request's temperature/top-p/top-k policy
(:func:`spec_probs`), so the rule compares the distributions the sampler would draw
from. The truncation mirrors the Triton sampling kernels exactly (keep ``x >= thr``,
boundary ties retained like flashinfer; top-k first, then nucleus with the target
scaled by the kept top-k mass, so no renormalized copy is needed) -- see
``kernel/triton/sampling.py`` (``_keep_tail``, ``_topp_fused``).

Driver protocol (the engine/scheduler wiring that consumes a verdict):
1. ``verdict = verify_step(...)`` (pure, no state).
2. ``finalize_spec_step(req, verdict, cache_manager)``: on a partial accept restores the
   pre-step GDN journal (the forward advanced the live slot past ``committed``), then
   rewrites the host tail with the accepted tokens and pairs the page release with the
   length rotation. Afterwards lengths/pages/host are authoritative through
   ``committed``; on a partial accept the GDN live state is stale (through the pre-step
   length) until the driver replays ``[replay_from, replay_to)`` as a continuation
   chunk -- the span's pages are still allocated, so the replay only refills state.
3. The drain must NOT append row spans for a finalized spec batch: the accepted tokens
   are already in ``input_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from freetoken.core import Req, SamplingParams


# Floor matching Sampler._prepare_rows_params: temperatures and top-p never reach exact
# zero through the sampler, so the verifier uses the same floors to form the same policy.
_MIN_T = 1e-6
_MIN_P = 1e-6


def spec_probs(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """The full sampling distribution for one position AFTER the request's policy.

    Greedy requests collapse to a one-hot at the argmax (the sampler draws argmax
    without forming probabilities, and the verify rule on one-hots reduces to comparing
    argmaxes). Otherwise: softmax over ``logits / T``, then the top-k mask (keep
    ``>=`` the k-th largest, ties retained), then nucleus over the masked mass with the
    target scaled by that mass (the kernel's exact staging), then renormalize.
    """
    flat = logits.flatten().float()
    vocab = flat.numel()
    if params.is_greedy:
        out = torch.zeros_like(flat)
        out[int(torch.argmax(flat).item())] = 1.0
        return out
    temperature = max(params.temperature, _MIN_T)
    probs = torch.softmax(flat / temperature, dim=-1)
    # verify runs on live forward logits (GPU); every mask must live with them.
    keep = torch.ones(vocab, dtype=torch.bool, device=flat.device)
    top_k = params.top_k if params.top_k >= 1 else vocab
    if top_k < vocab:
        # keep x >= the k-th largest (boundary ties retained, like the kernel).
        kth = torch.topk(probs, top_k).values.min()
        keep &= probs >= kth
    top_p = min(max(params.top_p, _MIN_P), 1.0)
    if top_p < 1.0:
        # Nucleus over the top-k-kept mass with the target scaled by that mass (the
        # kernel never renormalizes between the stages): smallest prefix with
        # cumsum >= top_p * kept_mass, ties retained.
        masked = torch.where(keep, probs, torch.zeros(()))
        mass = float(masked.sum().item())
        target = top_p * mass
        ordered, _ = torch.sort(masked, descending=True)
        reached = ordered.cumsum(-1) >= target
        if not bool(reached.any().item()):
            pass  # fp rounding pushed the target past the total mass: keep the bracket
        else:
            thr = ordered[int(torch.argmax(reached.int()).item())]
            keep &= masked >= thr
    out = torch.where(keep, probs, torch.zeros(()))
    total = float(out.sum().item())
    if total <= 0.0:
        raise ValueError("speculative policy truncated the whole vocabulary")
    return out / total


@dataclass
class SpecVerdict:
    """The outcome of one verification pass, as token data (positions base_len+1 ...)."""

    accepted: list[int] = field(default_factory=list)
    n_accepted: int = 0  # drafts accepted (0..d); len(accepted) == n_accepted + 1
    bonus: bool = False  # the trailing token is a bonus (all drafts accepted), else a resample
    committed: int = 0  # base_len + len(accepted): authoritative length after finalize
    replay_from: int = 0  # accepted-but-unplayed GDN span for the driver to replay ...
    replay_to: int = 0  # ... as a continuation chunk ([from, to); empty when all accepted)

    @property
    def all_accepted(self) -> bool:
        return self.bonus

    @property
    def needs_replay(self) -> bool:
        return self.replay_to > self.replay_from


def _verify_greedy(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
    base_len: int,
) -> SpecVerdict:
    """Greedy verification without distributions (the fast path for the common case).

    The greedy target policy is argmax, so a draft accepts iff it equals the target
    argmax at its position and the first mismatch resamples to that argmax -- which is
    exactly what the general rule reduces to (one-hot ``p`` gives accept probability
    1/0, and ``norm(max(0, p - q))`` is the target one-hot). Comparing ids skips the
    per-token softmax + full-vocab one-hot pair of the general path; both id vectors
    come back in one transfer.
    """
    d = int(draft_tokens.numel())
    ids = torch.cat([draft_tokens, target_logits.argmax(dim=-1)]).tolist()  # one transfer
    drafts, argmax = ids[:d], ids[d:]
    n = 0
    while n < d and drafts[n] == argmax[n]:
        n += 1
    if n == d:
        accepted = drafts + [argmax[d]]
        return SpecVerdict(
            accepted=accepted, n_accepted=d, bonus=True,
            committed=base_len + len(accepted), replay_from=base_len, replay_to=base_len,
        )
    accepted = drafts[:n] + [argmax[n]]
    committed = base_len + len(accepted)
    return SpecVerdict(
        accepted=accepted, n_accepted=n, bonus=False, committed=committed,
        replay_from=base_len, replay_to=committed,
    )


def verify_step(
    draft_tokens: torch.Tensor,
    draft_logits: torch.Tensor | None,
    target_logits: torch.Tensor,
    params: SamplingParams,
    base_len: int,
    *,
    generator: torch.Generator | None = None,
) -> SpecVerdict:
    """Accept or reject ``draft_tokens`` against the target distributions.

    ``draft_logits`` is ``[d, V]`` (row ``i`` forms ``q_{i+1}``) for stochastic
    verification; a deterministic proposal (prompt look-up) passes ``None`` -- the
    greedy fast path compares ids and never reads a draft distribution. ``target_logits``
    is ``[d+1, V]`` (row ``i`` forms ``p_{i+1}``, row ``d`` the bonus distribution).
    Uniforms and draws come from ``generator`` when given (seed it for deterministic
    tests), else the global RNG.
    """
    d = int(draft_tokens.numel())
    vocab = int(target_logits.shape[-1])
    assert target_logits.shape == (d + 1, vocab), (tuple(target_logits.shape), d, vocab)
    if params.is_greedy:
        return _verify_greedy(draft_tokens, target_logits, base_len)
    assert draft_logits is not None and draft_logits.shape == (d, vocab), (
        None if draft_logits is None else tuple(draft_logits.shape), d, vocab)

    def rand() -> float:
        return float(torch.rand((), generator=generator).item())

    accepted: list[int] = []
    for i in range(d):
        p = spec_probs(target_logits[i], params)
        q = spec_probs(draft_logits[i], params)
        x = int(draft_tokens[i].item())
        px, qx = float(p[x].item()), float(q[x].item())
        # The standard accept probability min(1, p/q); a draft outside q's support
        # (impossible through the sampler, reachable in tests) accepts iff p allows it.
        alpha = min(1.0, px / qx) if qx > 0.0 else (1.0 if px > 0.0 else 0.0)
        if rand() < alpha:
            accepted.append(x)
            continue
        # Rejection: resample from norm(max(0, p - q)) and stop. p != q here (equal
        # distributions accept with probability 1, and rand() < 1 always), so the
        # residual has mass; the p fallback is unreachable defense-in-depth.
        residual = (p - q).clamp_(min=0.0)
        if float(residual.sum().item()) <= 0.0:
            residual = p
        accepted.append(
            int(torch.multinomial(residual / residual.sum(), 1, generator=generator).item())
        )
        n = len(accepted) - 1
        committed = base_len + len(accepted)
        return SpecVerdict(
            accepted=accepted, n_accepted=n, bonus=False, committed=committed,
            replay_from=base_len, replay_to=committed,
        )
    bonus = spec_probs(target_logits[d], params)
    if params.is_greedy:
        accepted.append(int(torch.argmax(bonus).item()))
    else:
        accepted.append(int(torch.multinomial(bonus, 1, generator=generator).item()))
    return SpecVerdict(
        accepted=accepted, n_accepted=d, bonus=True,
        committed=base_len + len(accepted), replay_from=base_len, replay_to=base_len,
    )


def finalize_spec_step(req: Req, verdict: SpecVerdict, cache_manager) -> None:
    """Apply a verdict's state, page, length and host updates (the drain's append is NOT
    repeated afterwards -- the accepted tokens are already in ``input_ids``).

    On a partial accept the live GDN state advanced past ``committed`` during the
    forward, so it is restored to the pre-step journal first; the driver then replays
    ``[verdict.replay_from, verdict.replay_to)`` as a continuation chunk before the next
    scoring forward. ``cache_manager`` is duck-typed (spec_restore/commit_spec).
    """
    if not verdict.all_accepted:
        cache_manager.spec_restore(req)
    req.accept_spec_tail(verdict.accepted, verdict.committed)
    cache_manager.commit_spec(req, verdict.committed)


@dataclass
class SpecAccounting:
    """Per-step draft acceptance counters for one MTP-enabled server run.

    ``record`` runs once per verified request-step: ``depth`` drafts were proposed,
    ``n_accepted`` of them survived verification (the trailing resample/bonus token is
    NOT a draft, so it is not counted on either side). The decode log line and the
    419.8 benchmark both read :meth:`snapshot` -- the dict shape is the contract::

        {"steps": int, "proposed": int, "accepted": int, "rate": float}

    ``rate`` is ``accepted / proposed`` (0.0 with no verified steps yet). The log
    fragment (:meth:`log_fragment`, ``""`` when idle so plain-decode lines are
    byte-identical) reads ``spec accept: {accepted}/{proposed} ({rate:.2f})``.
    """

    steps: int = 0
    proposed: int = 0
    accepted: int = 0

    def record(self, n_accepted: int, depth: int) -> None:
        assert 0 <= n_accepted <= depth, (n_accepted, depth)
        self.steps += 1
        self.proposed += depth
        self.accepted += n_accepted

    @property
    def rate(self) -> float:
        return self.accepted / self.proposed if self.proposed > 0 else 0.0

    def snapshot(self) -> dict:
        return {
            "steps": self.steps,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "rate": self.rate,
        }

    def log_fragment(self) -> str:
        if self.steps == 0:
            return ""
        return f"spec accept: {self.accepted}/{self.proposed} ({self.rate:.2f})"


__all__ = [
    "SpecAccounting",
    "SpecVerdict",
    "finalize_spec_step",
    "spec_probs",
    "verify_step",
]
