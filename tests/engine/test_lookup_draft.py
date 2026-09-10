"""Draft-free speculation (prompt look-up): the drafter and its scheduler wiring.

The look-up drafter proposes the continuation of the most recent earlier occurrence of
the request's token suffix; the same exact verify path as MTP decides. These tests pin
the suffix matching, the incremental index, and the scheduler's placement of the
stashed drafts (the schedule-time hook must agree with the rows the step reserved).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import SamplingParams
from freetoken.engine.lookup import LookupDrafter


def _req(uid: int, ids, table_idx: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        uid=uid, table_idx=table_idx,
        input_ids=torch.tensor(ids, dtype=torch.int64),
        sampling_params=SamplingParams(temperature=0.0),
        cached_len=len(ids) - 1, spec_depth=0,
    )


# ======================================================================================
# Drafter: matching and proposals
# ======================================================================================


def test_plan_proposes_the_earlier_continuation():
    drafter = LookupDrafter(max_ngram=6)
    req = _req(1, [1, 2, 3, 4, 5, 6, 7, 1, 2, 3])
    assert drafter.plan(req, 3, req.input_ids[-1].item()) == 3
    assert drafter.take(1, 3) == [4, 5, 6]


def test_plan_takes_the_longest_match_first():
    # (1, 2, 3) earlier ends at position 2 (continuation 4, 5); the shorter (2, 3)
    # also occurs later with a different continuation (9, 9) -- the longest match wins
    drafter = LookupDrafter(max_ngram=6)
    req = _req(1, [1, 2, 3, 4, 5, 2, 3, 9, 9, 1, 2, 3])
    assert drafter.plan(req, 2, req.input_ids[-1].item()) == 2
    assert drafter.take(1, 2) == [4, 5]


def test_plan_returns_zero_without_a_match():
    drafter = LookupDrafter(max_ngram=6)
    req = _req(1, [1, 2, 3, 4, 5, 6, 7, 8])
    assert drafter.plan(req, 4, req.input_ids[-1].item()) == 0
    assert drafter.take(1, 4) == []


def test_min_ngram_rejects_weak_matches():
    drafter = LookupDrafter(max_ngram=4, min_ngram=3)
    # only the pair (4, 7) and the single token 7 repeat; both below min_ngram
    req = _req(1, [1, 2, 3, 4, 7, 4, 7])
    assert drafter.plan(req, 4, req.input_ids[-1].item()) == 0


def test_plan_clamps_to_the_available_continuation():
    drafter = LookupDrafter(max_ngram=4)
    # the earlier occurrence of (1, 2) sits two tokens before the tip: only [1, 2]
    # remains before the sequence end, fewer than the requested depth
    req = _req(1, [7, 8, 1, 2, 1, 2])
    assert drafter.plan(req, 4, req.input_ids[-1].item()) == 2
    assert drafter.take(1, 4) == [1, 2]


def test_index_syncs_incrementally_across_steps():
    drafter = LookupDrafter(max_ngram=4)
    req = _req(1, [1, 2, 3, 4, 5, 6])
    assert drafter.plan(req, 3, req.input_ids[-1].item()) == 0
    req.input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3])
    req.cached_len = 10
    assert drafter.plan(req, 3, req.input_ids[-1].item()) == 3
    assert drafter.take(1, 3) == [4, 5, 6]


def test_propose_aligns_when_the_anchor_is_not_yet_indexed():
    """Overlap: a plain step's sampled anchor lives in the pool, not yet in input_ids.
    The suffix must be built with the anchor as its tip, so the proposals predict the
    positions AFTER the anchor (an off-by-one here made every draft reject live)."""
    drafter = LookupDrafter(max_ngram=6)
    req = _req(1, [1, 2, 3, 4, 5, 6, 7, 1, 2])  # anchor 3 is not in input_ids yet
    assert drafter.plan(req, 3, anchor_token=3) == 3
    assert drafter.take(1, 3) == [4, 5, 6]


def test_take_is_single_shot_and_drop_clears_state():
    drafter = LookupDrafter(max_ngram=4)
    req = _req(1, [1, 2, 3, 1, 2, 3])
    assert drafter.plan(req, 2, req.input_ids[-1].item()) == 2
    assert drafter.take(1, 2) == [1, 2]
    assert drafter.take(1, 2) == []  # consumed
    drafter.drop(1)
    assert drafter.plan(req, 2, req.input_ids[-1].item()) == 2  # rebuilds from scratch
    assert drafter.take(1, 2) == [1, 2]


# ======================================================================================
# Scheduler wiring: plan at schedule time, place at forward time
# ======================================================================================


def _lookup_sched(vocab: int = 16, lookup_draft: int = 3) -> SimpleNamespace:
    from freetoken.scheduler.scheduler import Scheduler

    sched = Scheduler.__new__(Scheduler)
    sched.config = SimpleNamespace(
        lookup_draft=lookup_draft, lookup_ngram=6,
        model_config=SimpleNamespace(vocab_size=vocab),
    )
    sched.device = torch.device("cpu")
    sched.token_pool = torch.zeros((2, 64), dtype=torch.int32)
    sched.lookup_drafter = LookupDrafter(max_ngram=6)
    return sched


def test_lookup_depth_plans_and_the_step_places_the_drafts():
    from freetoken.scheduler.scheduler import Scheduler

    sched = _lookup_sched()
    req = _req(1, [1, 2, 3, 4, 5, 6, 7, 1, 2, 3], table_idx=0)
    sched.token_pool[0, req.cached_len] = 3  # the anchor, not yet in input_ids
    depth = Scheduler._lookup_depth(sched, req)
    assert depth == 3
    req.spec_depth = depth
    batch = SimpleNamespace(reqs=[req])
    state = Scheduler._open_spec_step(sched, batch)
    base_len, tokens, logits = state[1]
    assert base_len == 9 and tokens.tolist() == [4, 5, 6]
    assert logits is None  # greedy: the id-compare fast path needs no distribution
    assert sched.token_pool[0, 10:13].tolist() == [4, 5, 6]  # base+1 .. base+depth


def test_lookup_step_builds_one_hot_logits_for_stochastic_requests():
    from freetoken.scheduler.scheduler import Scheduler

    sched = _lookup_sched()
    req = _req(1, [1, 2, 3, 4, 5, 6, 7, 1, 2, 3], table_idx=0)
    sched.token_pool[0, req.cached_len] = 3
    req.sampling_params = SamplingParams(temperature=1.0)
    req.spec_depth = Scheduler._lookup_depth(sched, req)
    batch = SimpleNamespace(reqs=[req])
    _base, tokens, logits = Scheduler._open_spec_step(sched, batch)[1]
    assert logits.shape == (3, 16)
    assert logits.argmax(dim=-1).tolist() == tokens.tolist()
    assert torch.equal(logits.max(dim=-1).values, torch.zeros(3))  # winner slot
