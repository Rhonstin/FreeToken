"""MTP speculative-decode driver: draft -> verify -> commit wiring.

Pure orchestration over the 419.2-419.5 machinery (the draft head, the scheduler's
multi-token frames, the GDN journal, the probability-correct verifier). Everything
CPU-runnable here is unit tested; the remaining GPU-only work is the hybrid
continuation replay (seam 4) and the live acceptance/throughput benchmark (419.8).

How the closed seams work:

1. Target hidden states: ``propose_drafts`` takes a ``step_fn`` closure that owns the
   previous target forward's 4-stream anchor residual. The engine retains per-request
   anchor rows on ``ForwardOutput.target_hidden_anchors`` (eager forwards only, so the
   plain graph path is untouched) and the scheduler's draft closure chains them across
   ``propose_drafts`` calls (anchor + previous draft residuals, see
   ``Qwen4ExpMTPHead.last_draft_residual``).
2. Draft attention KV: the production ``step_fn`` runs the draft layers
   (``draft_layer_ids``: MoE-bank/QSA-slot ids ``num_moe_layers + i``) whose KV pages
   and QSA slots the target-sized pools did not provision. The pool now extends its
   dense slot map with ``draft_qsa_layer_ids`` (same order the backend maps them)
   and widens the pending ring by the draft depth
   (``qsa_ring_capacity_for_depth``); the draft batch reuses the step's reserved
   pages, so no persistent draft KV is needed -- only the per-call slots.

GDN note (from 419.4): exact hybrid scoring runs as a continuation extend -- parallel
same-slot decode rows race in the GDN decode kernel. The scheduled spec-rows forward
still advances the live slot through every draft row (journaled pre-forward by
``spec_snapshot``), so a partial accept restores the journal and the driver replays
``[verdict.replay_from, verdict.replay_to)`` as a continuation chunk before the next
scoring forward. ``free_token_tail`` is single-shot per commit (a second call would
double-free); ``commit_spec`` already pairs the release with the length rotation, so
the driver never calls either directly -- only ``finalize_spec_step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from freetoken.utils import init_logger

from .spec import SpecAccounting, SpecVerdict, finalize_spec_step, verify_step

if TYPE_CHECKING:
    from freetoken.core import Req

logger = init_logger(__name__)

__all__ = [
    "AdaptiveDepthPolicy",
    "MtpConfig",
    "attach_mtp_draft_cache",
    "build_mtp_draft_banks",
    "clamp_spec_depth",
    "collect_mtp_expert_pieces",
    "draft_forward",
    "draft_moe_layers",
    "draft_qsa_layer_ids",
    "draft_write_positions",
    "make_spec_depth_fn",
    "mtp_bank_layer_ids",
    "propose_drafts",
    "qsa_ring_capacity_for_depth",
    "spec_anchor_hidden",
    "verify_and_finalize",
]


@dataclass(frozen=True)
class MtpConfig:
    """Resolved MTP run config: ``depth`` drafts per step (0 = plain decode, inert),
    ``adaptive`` lets each request's depth track its own acceptance (``--speculative-adaptive``)."""

    depth: int = 0
    adaptive: bool = False

    @property
    def enabled(self) -> bool:
        return self.depth > 0

    @classmethod
    def from_config(cls, cfg) -> "MtpConfig":
        """Read off any engine/scheduler config object (duck-typed: ``mtp_depth`` / ``mtp_adaptive``)."""
        return cls(
            depth=int(getattr(cfg, "mtp_depth", 0) or 0),
            adaptive=bool(getattr(cfg, "mtp_adaptive", False)),
        )


def clamp_spec_depth(requested: int, cached_len: int, max_device_len: int) -> int:
    """Draft rows that fit this step: ``requested`` clamped to the output budget.

    The forwarded frame is anchor + drafts at positions ``[cached_len, cached_len +
    1 + depth]``, and a fully-accepted step commits every row's continuation -- the
    last one at ``cached_len + 1 + depth``. The commit's host rewrite stores that
    token at its index, so the deepest legal commit is index ``max_device_len - 1``:
    at most ``max_device_len - cached_len - 2`` drafts (one slot stays reserved for
    the row immediately past the frame). Negatives clamp to 0 (plain row). Same
    formula the DecodeManager applies at schedule time -- the driver clamps early
    so drafting never proposes tokens the schedule would drop.
    """
    return max(0, min(int(requested), max_device_len - cached_len - 2))


class AdaptiveDepthPolicy:
    """Per-request draft depth tracking recent acceptance (``--speculative-adaptive``).

    Uniform ``--mtp-depth`` assumes every request accepts alike; under adaptive each
    uid starts at ``max_depth`` and steps +-1 per verified step: a fully-accepted step
    grows (up to the max), an all-rejected step shrinks (down to 0, a plain row for
    that request), a partial accept holds. A depth-0 step carries no signal (nothing
    was proposed), so it never updates.
    """

    def __init__(self, max_depth: int) -> None:
        assert max_depth >= 0, max_depth
        self._max = int(max_depth)
        self._depth: dict[int, int] = {}

    @property
    def max_depth(self) -> int:
        return self._max

    def depth_for(self, uid: int) -> int:
        return self._depth.get(uid, self._max)

    def record(self, uid: int, n_accepted: int, depth: int) -> int:
        cur = self._depth.get(uid, self._max)
        if depth <= 0:
            return cur
        if n_accepted >= depth:
            nxt = min(self._max, cur + 1)
        elif n_accepted == 0:
            nxt = max(0, cur - 1)
        else:
            nxt = cur
        self._depth[uid] = nxt
        return nxt


def make_spec_depth_fn(
    depth: int, *, adaptive: bool = False, policy: AdaptiveDepthPolicy | None = None,
):
    """The ``DecodeManager.spec_depth_fn``, or None to keep plain decode byte-identical.

    None (not a zero-fn) when ``depth <= 0``: the manager then skips ``reserve_spec``
    entirely and the step frame is untouched. Otherwise each request's want is clamped
    to its output budget; under ``adaptive`` the want comes from the per-uid policy
    (created here when the caller passes none, exposed as ``fn.policy`` for the
    verify path to record into).
    """
    if depth <= 0:
        if adaptive:
            logger.warning("--speculative-adaptive without --mtp-depth is inert")
        return None
    if adaptive and policy is None:
        policy = AdaptiveDepthPolicy(depth)

    def fn(req) -> int:
        want = policy.depth_for(req.uid) if policy is not None else depth
        return clamp_spec_depth(want, req.cached_len, req.max_device_len)

    fn.policy = policy  # type: ignore[attr-defined]
    return fn


def draft_write_positions(cached_len: int, depth: int) -> list[int]:
    """Token-pool write slots for this step's drafts: ``cached_len + 1 + spec_index``.

    The anchor's continuation and each draft row's continuation land one past each
    input row (see ``_make_write_tuple``); the drafts themselves are the inputs at
    ``[cached_len + 1, cached_len + 1 + depth)``, one past the reserved frame exactly
    the way a plain decode stages its next-step anchor.
    """
    return [cached_len + 1 + i for i in range(max(0, depth))]


def write_draft_tokens(token_pool: torch.Tensor, table_idx: int, positions, draft_tokens: torch.Tensor) -> None:
    """Place draft ids into the device token pool BEFORE the verify forward gathers inputs.

    The scheduler calls this after page allocation with the positions from
    :func:`draft_write_positions`; the forward's ``token_pool[input_mapping]`` gather
    then reads the drafts as the draft rows' inputs. Device-agnostic (CPU tests pass a
    CPU pool); the in-place write must stay on the caller's stream.
    """
    pos = torch.as_tensor(list(positions), dtype=torch.int64, device=token_pool.device)
    assert len(pos) == int(draft_tokens.numel()), (len(pos), int(draft_tokens.numel()))
    token_pool[int(table_idx), pos] = draft_tokens.to(
        device=token_pool.device, dtype=token_pool.dtype)


def propose_drafts(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    seed_token: int,
    depth: int,
    *,
    choose: Callable[[torch.Tensor], int] | None = None,
    dtype: torch.dtype = torch.int64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively draft ``depth`` tokens, returning ``(tokens [d], logits [d, V])``.

    ``step_fn(prefix_tokens) -> logits [T, V]`` evaluates the draft stack over the
    current draft prefix (embed + MTP head + shared lm_head inside; the production
    closure owns the previous target forward's anchor hidden and the draft batch/KV
    -- GPU-only -- while tests pass a stub returning canned logits). Each iteration
    appends the chosen token and re-evaluates the whole few-token prefix, so the
    driver holds no persistent draft KV. ``choose`` picks the next token from one
    logits row (default greedy argmax; stochastic drafting passes its own sampler);
    row ``i`` of the returned logits is the distribution the token at draft position
    ``i`` was drawn from, matching ``verify_step``'s ``draft_logits`` contract.
    """
    if choose is None:
        def choose(row: torch.Tensor) -> int:
            return int(torch.argmax(row).item())
    tokens: list[int] = []
    rows: list[torch.Tensor] = []
    prefix = torch.as_tensor([seed_token], dtype=dtype, device=device)
    for _ in range(max(0, depth)):
        logits = step_fn(prefix)
        assert logits.dim() == 2 and logits.shape[0] == prefix.numel(), (
            tuple(logits.shape), prefix.numel())
        row = logits[-1]
        tokens.append(choose(row))
        rows.append(row)
        prefix = torch.cat([prefix, torch.as_tensor(
            [tokens[-1]], dtype=dtype, device=device)])
    if not tokens:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, 0, dtype=torch.float32, device=device),
        )
    return (
        torch.as_tensor(tokens, dtype=torch.int64, device=device),
        torch.stack(rows),
    )


def spec_anchor_hidden(model_hidden: torch.Tensor | None, batch) -> dict | None:
    """Per-request anchor spans of the target's 4-stream residual, keyed by uid.

    The draft closure pairs ``(h_{p-1}, x_p)`` -- the hidden of the position BEFORE
    the anchor token, matching how the head is trained (the reference shifts the
    target hidden right by one row). Which row that is depends on how many drafts
    the verify pass will accept, so this retains the request's WHOLE span of rows
    plus its base position: the draft closure picks ``anchor_pos - 1 - base`` once
    the committed length is known. ``model_hidden`` is the model's stashed
    post-forward residual (None when the model family does not expose one); the
    spans are views, kept alive one step by the scheduler. Pure (CPU-testable);
    ``batch`` is duck-typed (spec_rows / padded_reqs / reqs).
    """
    if model_hidden is None:
        return None
    anchors: dict = {}
    if batch.spec_active:
        rows = batch.spec_rows
        assert model_hidden.shape[0] == len(rows), (tuple(model_hidden.shape), len(rows))
        i = 0
        while i < len(rows):
            req = rows[i][0]
            span = 1
            while i + span < len(rows) and rows[i + span][0] is req:
                span += 1
            if req.uid >= 0:
                anchors[req.uid] = (req.cached_len, model_hidden[i : i + span])
            i += span
        return anchors
    reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
    if batch.is_prefill:
        offset = 0
        for req in reqs:
            n = req.extend_len
            if req.uid >= 0:
                anchors[req.uid] = (req.cached_len, model_hidden[offset : offset + n])
            offset += n
        assert offset == model_hidden.shape[0], (offset, tuple(model_hidden.shape))
        return anchors
    assert model_hidden.shape[0] == len(reqs), (tuple(model_hidden.shape), len(reqs))
    for row, req in enumerate(reqs):
        if req.uid >= 0:
            anchors[req.uid] = (req.cached_len, model_hidden[row : row + 1])
    return anchors


def draft_forward(model, prefix_tokens: torch.Tensor, target_hidden: torch.Tensor, batch) -> torch.Tensor:
    """Production single draft-stack evaluation over a draft prefix (GPU-only).

    Embeds ``prefix_tokens`` with the target table, runs the MTP head against the
    chained ``target_hidden`` rows (anchor residual + previous draft residuals, owned
    by the scheduler's draft closure), projects with the shared lm_head, returns
    ``[T, V]`` logits. The draft batch runs entered as the global batch (the MoE
    layers branch on ``ctx.batch.is_prefill``); its KV lands in the draft layers'
    pool slots, which the factory provisions past the target's layers.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("MTP draft forward needs a GPU (torch.cuda.is_available() is False)")
    from freetoken.core import get_global_ctx

    mtp_head = getattr(model, "mtp", None)
    if mtp_head is None:
        raise RuntimeError("MTP draft forward needs a model with an MTP head built")
    t = int(prefix_tokens.numel())
    assert target_hidden.shape[0] == t, (tuple(target_hidden.shape), t)
    with get_global_ctx().forward_batch(batch):
        embeds = model.model.embed_tokens.forward(prefix_tokens)
        hidden = mtp_head.forward(embeds, target_hidden, batch)
        # all_rows: the draft batch runs prefill-phase (chunk semantics for the
        # draft attention/MoE), but every draft position's distribution is needed --
        # the head's default prefill slicing would keep only the last row.
        return model.lm_head.forward(hidden, all_rows=True)


def verify_and_finalize(
    req: "Req",
    *,
    draft_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    base_len: int,
    cache_manager,
    accounting: SpecAccounting,
    policy: AdaptiveDepthPolicy | None = None,
    generator: torch.Generator | None = None,
) -> SpecVerdict | None:
    """One request's verify pass after the scoring forward; None when nothing was proposed.

    ``base_len`` is the pre-forward ``cached_len`` (the engine's ``complete_n`` already
    advanced lengths by the time this runs post-forward). Runs ``verify_step`` over the
    position-indexed tensors -- for a decode-rows forward, target row ``i`` holds
    ``p_{i+1}`` since row ``i`` consumes position ``base_len + i`` -- then
    ``finalize_spec_step`` (journal restore on partial accept, host-tail rewrite, the
    single-shot page release paired with the length rotation), then records the
    acceptance counters (and the adaptive policy). Draft tensors come from
    :func:`propose_drafts`; target rows are sliced from the scoring forward's logits.
    """
    d = int(draft_tokens.numel())
    if d == 0:
        return None
    # The scoring forward already advanced lengths by the full span (complete_n runs
    # inside forward_batch, before this post-forward pass). Rewind to the scheduled
    # reservation frame first: finalize's host rewrite, page release and rotation are
    # built and tested against it, and the post-advance device overshoots the
    # reservation end by one (a page-boundary free-list corruption). spec_depth is
    # untouched by completion; the proposal never exceeds it, so the frame is exact.
    assert 0 <= d <= req.spec_depth, (d, req.spec_depth)
    req.cached_len = base_len
    req.device_len = base_len + 1 + req.spec_depth
    verdict = verify_step(
        draft_tokens, draft_logits, target_logits, req.sampling_params,
        base_len, generator=generator,
    )
    finalize_spec_step(req, verdict, cache_manager)
    accounting.record(verdict.n_accepted, d)
    if policy is not None:
        policy.record(req.uid, verdict.n_accepted, d)
    return verdict


def mtp_bank_layer_ids(model_config, num_mtp_layers: int) -> list[int]:
    """Offload-bank layer ids for the draft MoE layers: past the target's MoE layers.

    Matches ``Qwen4ExpMTPHead.draft_layer_ids`` and ``iter_mtp_expert_pieces`` so the
    target's expert method packs both into identical banks for one offload cache.
    Duck-typed (needs only ``num_moe_layers``) for CPU tests.
    """
    return [model_config.num_moe_layers + i for i in range(max(0, num_mtp_layers))]


def draft_qsa_layer_ids(model_config) -> list[int]:
    """QSA slot ids for the draft layers: the same numbers, in the same order.

    The KV pool and the QSA backend extend their dense slot maps with exactly this
    list (appended after the target's QSA layer ids), so both sides agree without
    sharing state. Empty when the model builds no draft head -- the pool and the
    backend then stay byte-identical to the plain path.
    """
    num_mtp_layers = int(getattr(getattr(model_config, "qwen4_args", None), "mtp_num_layers", 0) or 0)
    if num_mtp_layers <= 0:
        return []
    return mtp_bank_layer_ids(model_config, num_mtp_layers)


def collect_mtp_expert_pieces(model_path: str, model_config) -> list:
    """Eager consumer of ``iter_mtp_expert_pieces``: ``(bank_layer_id, e0, e1, pieces)``.

    The MTP head stores one fused bf16 tensor per role per layer; this slices the
    per-expert gate/up halves out exactly like the bf16 bank pack expects, keyed by
    the draft bank layer ids. The GPU box feeds these into ``build_expert_banks``
    through the draft layers' own (unquantized) method -- see
    :func:`build_mtp_draft_banks` -- into a dedicated bf16 draft cache (one cache
    holds one bank schema, so the NVFP4 target cache cannot serve these).
    """
    from freetoken.models.qwen4_exp.weight import iter_mtp_expert_pieces

    return list(iter_mtp_expert_pieces(model_path, model_config))


def draft_moe_layers(model) -> list:
    """The MTP head's offload MoE layers in draft order (empty when no head is built).

    Walks only the ``mtp`` subtree, so target layers never mix in; order follows the
    head's ``OPList`` (draft layer ``i`` is element ``i``).
    """
    from freetoken.moe.offload_cache import iter_offload_moe_layers

    mtp_head = getattr(model, "mtp", None)
    if mtp_head is None:
        return []
    return list(iter_offload_moe_layers(mtp_head))


def build_mtp_draft_banks(model, model_path: str, model_config, num_mtp_layers: int,
                          *, device, dummy: bool = False):
    """Pack the checkpoint's stacked MTP experts into bf16 host banks.

    The pack method is the draft layers' own quant method (unquantized bf16 on the
    shipping checkpoint, whose ``mtp.*`` the quant ignore list covers) -- by
    construction the exact layout the layers' kernels read. Pieces arrive with global
    bank ids (``num_moe_layers + i``); they are remapped to local draft rows
    (``0..num_mtp_layers``) for the dedicated draft cache. ``dummy`` fills finite
    random banks without touching the checkpoint (``--use-dummy-weight``).
    """
    from freetoken.moe.expert_banks import build_expert_banks

    layers = draft_moe_layers(model)
    if len(layers) != num_mtp_layers or num_mtp_layers <= 0:
        raise ValueError(
            f"MTP draft banks need {num_mtp_layers} draft MoE layers, "
            f"found {len(layers)} under model.mtp")
    for layer in layers:
        if getattr(layer, "quant_method", None) is None:
            raise ValueError(
                "MTP draft MoE layers carry no quant method (need the offload-family "
                "--moe-strategy so they build bank-backed experts)")
    method = layers[0].quant_method
    base = model_config.num_moe_layers

    def _local_pieces():
        for bank_layer_id, e0, e1, piece in collect_mtp_expert_pieces(
                model_path, model_config):
            yield bank_layer_id - base, e0, e1, piece

    return build_expert_banks(
        method, num_mtp_layers, _local_pieces(), device=device, dummy=dummy)


def attach_mtp_draft_cache(model, cache) -> list:
    """Bind the draft banks cache to the draft MoE layers, returning them.

    Draft MoE layers are constructed with global ids (``num_moe_layers + i``, shared
    with the QSA slot numbering); the dedicated draft cache only holds the
    ``num_mtp_layers`` draft rows, so each layer is rebound to its local row first.
    The attention layer ids (separate objects on the decoder layers) are untouched.
    """
    layers = draft_moe_layers(model)
    assert len(layers) == cache.num_layers, (len(layers), cache.num_layers)
    for local_id, layer in enumerate(layers):
        layer.layer_id = local_id
        layer.offload_cache = cache
    return layers


def qsa_ring_capacity_for_depth(index_ratio: int, depth: int) -> int:
    """QSA pending-ring depth widened by the draft depth (vLLM sizing).

    The pool-sizing hook for MTP: construct the QSA pool with
    ``ring_capacity=qsa_ring_capacity_for_depth(ratio, mtp_depth)`` (see
    ``create_kv_pool``'s QSA branch) so a spec step's pending index keys fit. Pure so
    the sizing stays CPU-testable; the construction change itself rides the GPU-box
    enablement (pools are GPU tensors).
    """
    from freetoken.kvcache.qsa_pool import QSAKVCache

    return QSAKVCache.ring_capacity_for(index_ratio, max(0, depth))
