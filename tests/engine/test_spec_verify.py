"""Probability-correct verification and sampling.

A wrong acceptance rule does not crash, it silently degrades quality -- so the core of
this file is distribution-equivalence: the accept/reject/resample loop must reproduce
the target distribution exactly, under every policy (plain, top-k, top-p, greedy).
Deterministic edge cases (identical/disjoint distributions) pin the rule without any
statistics; the verdict-span and finalize tests pin the driver contract.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.engine.spec import (
    SpecVerdict,
    finalize_spec_step,
    spec_probs,
    verify_step,
)

V = 6


def _params(**overrides) -> SamplingParams:
    base = dict(temperature=1.0, top_k=-1, top_p=1.0, max_tokens=64)
    base.update(overrides)
    return SamplingParams(**base)


def _logits(seed: int, rows: int, scale: float = 2.0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, V, generator=gen) * scale


def _hist(samples: torch.Tensor) -> torch.Tensor:
    return torch.bincount(samples, minlength=V).float() / samples.numel()


# ======================================================================================
# spec_probs: the policy AFTER which p and q are compared
# ======================================================================================


def test_greedy_collapses_to_a_one_hot_argmax():
    logits = torch.tensor([0.1, 2.0, 1.0, -0.5, 0.3, 0.0])
    assert torch.equal(
        spec_probs(logits, _params(temperature=0.0)), torch.tensor([0, 1, 0, 0, 0, 0.0]))
    # top_k == 1 is greedy too, whatever the temperature
    assert torch.equal(
        spec_probs(logits, _params(temperature=1.0, top_k=1)),
        torch.tensor([0, 1, 0, 0, 0, 0.0]))


def test_temperature_matches_softmax_exactly():
    logits = _logits(3, 1)[0]
    assert torch.allclose(
        spec_probs(logits, _params(temperature=0.7)),
        torch.softmax(logits / 0.7, dim=-1), atol=1e-6)


def test_top_k_keeps_the_k_largest_and_retains_ties():
    probs = torch.tensor([0.4, 0.2, 0.2, 0.2, 0.0, 0.0])
    logits = torch.log(probs + 1e-30)
    out = spec_probs(logits, _params(top_k=2))
    # the 2nd-largest value is 0.2 with a three-way tie: every tie stays, like the kernel
    assert torch.equal(out > 0, torch.tensor([True] * 4 + [False] * 2))
    assert float(out.sum()) == pytest.approx(1.0)
    assert float(out[0]) == pytest.approx(0.4)


def test_top_k_exact_cut_on_random_logits():
    logits = _logits(5, 1)[0]
    out = spec_probs(logits, _params(top_k=3))
    ref = torch.softmax(logits, dim=-1)
    thr = torch.topk(ref, 3).values.min()
    assert torch.equal(out > 0, ref >= thr)
    assert float(out.sum()) == pytest.approx(1.0)


def test_top_k_nonpositive_keeps_everything():
    logits = _logits(6, 1)[0]
    for k in (-1, 0):
        assert torch.allclose(
            spec_probs(logits, _params(top_k=k)), torch.softmax(logits, -1), atol=1e-6)


def test_top_p_hand_case():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05, 0.0, 0.0]) + 1e-30)
    out = spec_probs(logits, _params(top_p=0.8))
    # smallest prefix reaching 0.8 is {0.5, 0.3}: renormalized [0.625, 0.375]
    assert torch.equal(out > 0, torch.tensor([True, True] + [False] * 4))
    assert float(out[0]) == pytest.approx(0.625)
    assert float(out[1]) == pytest.approx(0.375)


def test_top_p_zero_keeps_only_the_argmax_set():
    logits = torch.tensor([0.1, 3.0, 3.0, 0.2, -1.0, 0.0])
    out = spec_probs(logits, _params(top_p=0.0))
    assert torch.equal(out > 0, torch.tensor([False, True, True, False, False, False]))
    assert float(out[1]) == pytest.approx(0.5)


def test_combined_top_k_top_p_matches_two_stage_staging():
    logits = _logits(7, 1)[0]
    out = spec_probs(logits, _params(top_k=4, top_p=0.7))
    ref = torch.softmax(logits, dim=-1)
    kth = torch.topk(ref, 4).values.min()
    kept = torch.where(ref >= kth, ref, torch.zeros(()))
    mass = float(kept.sum())
    ordered, _ = torch.sort(kept, descending=True)
    n = int((ordered.cumsum(-1) >= 0.7 * mass).int().argmax())
    thr = ordered[n]
    want = torch.where(kept >= thr, kept, torch.zeros(()))
    want = want / want.sum()
    assert torch.allclose(out, want, atol=1e-6)


# ======================================================================================
# verify_step: deterministic edges of the acceptance rule
# ======================================================================================


def _onehot(idx: int) -> torch.Tensor:
    t = torch.full((V,), -1e30)
    t[idx] = 0.0
    return t


def test_identical_distributions_accept_everything_deterministically():
    d, base = 3, 20
    draft = torch.tensor([1, 4, 2])
    logits = _logits(11, d)  # q == p row-wise: alpha == 1, and rand() < 1 always
    target = torch.cat([logits, _logits(12, 1)])
    verdict = verify_step(draft, logits, target, _params(), base,
                          generator=torch.Generator().manual_seed(0))
    assert verdict.accepted == [1, 4, 2, verdict.accepted[-1]]
    assert (verdict.n_accepted, verdict.bonus) == (d, True)
    assert verdict.committed == base + d + 1
    assert not verdict.needs_replay


def test_disjoint_distributions_reject_and_resample():
    base = 20
    draft = torch.tensor([3])
    q = _onehot(3).unsqueeze(0)
    p = torch.cat([_onehot(1).unsqueeze(0), _onehot(4).unsqueeze(0)])
    verdict = verify_step(draft, q, p, _params(), base,
                          generator=torch.Generator().manual_seed(0))
    # p(3) == 0 -> alpha 0 -> reject; residual is one-hot at 1 -> resample is 1
    assert verdict.accepted == [1]
    assert (verdict.n_accepted, verdict.bonus) == (0, False)
    assert verdict.committed == base + 1
    assert (verdict.replay_from, verdict.replay_to) == (base, base + 1)


def test_accept_then_reject_stops_and_reports_the_span():
    base = 20
    draft = torch.tensor([1, 3])
    shared = _logits(21, 1)
    q = torch.cat([shared, _onehot(3).unsqueeze(0)])
    p = torch.cat([shared, _onehot(5).unsqueeze(0), _onehot(0).unsqueeze(0)])
    verdict = verify_step(draft, q, p, _params(), base,
                          generator=torch.Generator().manual_seed(0))
    assert verdict.accepted == [1, 5]
    assert (verdict.n_accepted, verdict.bonus) == (1, False)
    assert verdict.committed == base + 2
    assert (verdict.replay_from, verdict.replay_to) == (base, base + 2)


def test_greedy_accepts_on_argmax_match_and_resamples_on_mismatch():
    base = 20
    greedy = _params(temperature=0.0)
    # match at row 0 (both argmax 2), mismatch at row 1 (draft 4, target 0)
    q = torch.stack([_onehot(2), _onehot(4)])
    p = torch.stack([_onehot(2), _onehot(0), _onehot(1)])
    verdict = verify_step(torch.tensor([2, 4]), q, p, greedy, base,
                          generator=torch.Generator().manual_seed(0))
    assert verdict.accepted == [2, 0]
    assert (verdict.n_accepted, verdict.bonus) == (1, False)
    assert verdict.committed == base + 2


def test_greedy_all_match_draws_the_argmax_bonus():
    base = 20
    greedy = _params(temperature=0.0)
    q = torch.stack([_onehot(2)])
    p = torch.stack([_onehot(2), _onehot(5)])
    verdict = verify_step(torch.tensor([2]), q, p, greedy, base,
                          generator=torch.Generator().manual_seed(0))
    assert verdict.accepted == [2, 5]
    assert verdict.bonus and not verdict.needs_replay


def test_zero_drafts_is_bonus_only():
    base = 20
    bonus_logits = _logits(30, 1)
    verdict = verify_step(
        torch.empty(0, dtype=torch.int64), torch.empty(0, V),
        bonus_logits, _params(), base, generator=torch.Generator().manual_seed(0))
    assert (verdict.n_accepted, verdict.bonus) == (0, True)
    assert len(verdict.accepted) == 1 and verdict.committed == base + 1


# ======================================================================================
# Distribution equivalence: the loop reproduces the target distribution
# ======================================================================================

N = 60000


def _trial_outputs(logits_p: torch.Tensor, logits_q: torch.Tensor,
                   params: SamplingParams, seed: int) -> torch.Tensor:
    """Single-position verify outcomes, vectorized over N trials (rejects resampled)."""
    gen = torch.Generator().manual_seed(seed)
    p, q = spec_probs(logits_p, params), spec_probs(logits_q, params)
    xs = torch.multinomial(q.expand(N, -1), 1, generator=gen).squeeze(1)
    u = torch.rand(N, generator=gen)
    alpha = torch.minimum(torch.ones(()), p[xs] / q[xs].clamp(min=1e-30))
    out = xs.clone()
    rej = torch.nonzero(u >= alpha).squeeze(1)
    if rej.numel():
        r = (p - q).clamp(min=0.0)
        r = r / r.sum()
        out[rej] = torch.multinomial(r.expand(rej.numel(), -1), 1, generator=gen).squeeze(1)
    return out


def _assert_same_distribution(out: torch.Tensor, want: torch.Tensor, tol: float = 0.03):
    got = _hist(out)
    assert (got - want).abs().sum() <= tol, (got.tolist(), want.tolist())


def test_loop_reproduces_the_target_plain_softmax():
    p, q = spec_probs(_logits(40, 1)[0], _params()), spec_probs(_logits(41, 1)[0], _params())
    _assert_same_distribution(_trial_outputs(_logits(40, 1)[0], _logits(41, 1)[0], _params(), 1), p)


def test_loop_reproduces_the_target_under_top_k():
    params = _params(top_k=3)
    lp, lq = _logits(42, 1)[0], _logits(43, 1)[0]
    _assert_same_distribution(_trial_outputs(lp, lq, params, 2), spec_probs(lp, params))


def test_loop_reproduces_the_target_under_top_p():
    params = _params(top_p=0.75)
    lp, lq = _logits(44, 1)[0], _logits(45, 1)[0]
    _assert_same_distribution(_trial_outputs(lp, lq, params, 3), spec_probs(lp, params))


def test_loop_reproduces_the_target_under_combined_policy():
    params = _params(temperature=0.8, top_k=4, top_p=0.9)
    lp, lq = _logits(46, 1)[0], _logits(47, 1)[0]
    _assert_same_distribution(_trial_outputs(lp, lq, params, 4), spec_probs(lp, params))


def test_bonus_matches_the_last_target_row():
    """With q == p everywhere, every draft accepts and the bonus must follow p_last."""
    gen = torch.Generator().manual_seed(5)
    lp = _logits(48, 1)[0]
    p = spec_probs(lp, _params())
    bonus = torch.multinomial(p.expand(N, -1), 1, generator=gen).squeeze(1)
    _assert_same_distribution(bonus, p)


def test_verify_step_bonus_path_matches_the_last_row_over_many_steps():
    """End-to-end through verify_step: identical distributions accept everything, so the
    trailing accepted token of each step follows the step's last target row."""
    gen = torch.Generator().manual_seed(6)
    last = _logits(300, 1)  # one fixed bonus distribution for every step
    outs = []
    for step in range(2000):
        g = torch.Generator().manual_seed(100 + step)
        logits = _logits(200 + step, 2)
        target = torch.cat([logits, last])
        draft = torch.multinomial(
            torch.softmax(logits, -1), 1, generator=g).squeeze(1)
        verdict = verify_step(draft, logits, target, _params(), 0, generator=g)
        assert verdict.bonus
        outs.append(verdict.accepted[-1])
    _assert_same_distribution(torch.tensor(outs), spec_probs(last[0], _params()),
                              tol=0.08)


# ======================================================================================
# finalize_spec_step: the driver contract (host, pages, lengths, journal)
# ======================================================================================


def _framed_req(prompt: int, output: int, cached: int, depth: int, table_idx: int = 0) -> Req:
    req = Req(
        input_ids=torch.arange(prompt, dtype=torch.int64),
        table_idx=table_idx,
        cached_len=cached,
        output_len=output,
        uid=7,
        sampling_params=SamplingParams(max_tokens=output),
        cache_handle=None,
    )
    req.cached_len, req.device_len, req.spec_depth = cached, cached + 1, 0
    req.reserve_spec(depth)
    return req


def test_finalize_partial_rewrites_the_host_tail_and_rotates():
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.table import TableManager

    page_size, num_pages = 1, 32
    tm = TableManager(2, torch.zeros((2, num_pages * page_size), dtype=torch.int32))
    cm = CacheManager(num_pages, page_size, tm.page_table, "radix")
    req = _framed_req(prompt=8, output=16, cached=7, depth=3)
    req.table_idx = tm.allocate()
    # frame inputs: last-fixed 7, drafts [70, 71, 72]; verdict keeps 70, resamples 99
    req._ids_buf[8:12] = torch.tensor([70, 71, 72, 0])
    req.input_ids = req._ids_buf[:12]
    cm.allocate_paged([req])
    free_before = len(cm.free_slots)

    verdict = SpecVerdict(accepted=[70, 99], n_accepted=1, bonus=False,
                          committed=9, replay_from=7, replay_to=9)
    finalize_spec_step(req, verdict, cm)

    assert req.input_ids.tolist() == list(range(8)) + [70, 99]
    assert (req.cached_len, req.device_len, req.spec_depth) == (9, 10, 0)
    # page_size 1: the two rejected tail slots return
    assert len(cm.free_slots) == free_before + 2


def test_finalize_full_acceptance_appends_the_bonus_in_place():
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.table import TableManager

    page_size, num_pages = 1, 32
    tm = TableManager(2, torch.zeros((2, num_pages * page_size), dtype=torch.int32))
    cm = CacheManager(num_pages, page_size, tm.page_table, "radix")
    req = _framed_req(prompt=8, output=16, cached=7, depth=2)
    req.table_idx = tm.allocate()
    req._ids_buf[8:11] = torch.tensor([70, 71, 0])
    req.input_ids = req._ids_buf[:11]
    cm.allocate_paged([req])
    free_before = len(cm.free_slots)

    verdict = SpecVerdict(accepted=[70, 71, 99], n_accepted=2, bonus=True,
                          committed=10, replay_from=7, replay_to=7)
    finalize_spec_step(req, verdict, cm)

    assert req.input_ids.tolist() == list(range(8)) + [70, 71, 99]
    assert (req.cached_len, req.device_len, req.spec_depth) == (10, 11, 0)
    assert len(cm.free_slots) == free_before  # nothing rejected


def test_finalize_partial_restores_the_gdn_journal():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    from freetoken.scheduler.cache import CacheManager

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=1, num_value_heads=1,
        key_head_dim=8, value_head_dim=8, conv_kernel_dim=4, output_gate="sigmoid")
    pool = LinearStatePool(group, 8, torch.float32, torch.device("cpu"))
    table = torch.zeros((2, 32 * 4), dtype=torch.int32)
    cm = CacheManager(32, 4, table, "hybrid_radix", linear_state_pool=pool)

    req = _framed_req(prompt=8, output=16, cached=7, depth=2)
    req.table_idx = 0
    req.linear_slot_idx = pool.alloc(1)[0]
    pool.recurrent_states[:, req.linear_slot_idx].fill_(3.0)
    cm.spec_snapshot(req)
    pool.recurrent_states[:, req.linear_slot_idx].fill_(-1.0)  # the forward's advance

    verdict = SpecVerdict(accepted=[70, 99], n_accepted=1, bonus=False,
                          committed=9, replay_from=7, replay_to=9)
    finalize_spec_step(req, verdict, cm)
    assert (pool.recurrent_states[:, req.linear_slot_idx] == 3.0).all()
    assert (req.cached_len, req.device_len) == (9, 10)


def test_finalize_full_acceptance_leaves_advanced_state_alone():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    from freetoken.scheduler.cache import CacheManager

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=1, num_value_heads=1,
        key_head_dim=8, value_head_dim=8, conv_kernel_dim=4, output_gate="sigmoid")
    pool = LinearStatePool(group, 8, torch.float32, torch.device("cpu"))
    table = torch.zeros((2, 32 * 4), dtype=torch.int32)
    cm = CacheManager(32, 4, table, "hybrid_radix", linear_state_pool=pool)

    req = _framed_req(prompt=8, output=16, cached=7, depth=2)
    req.table_idx = 0
    req.linear_slot_idx = pool.alloc(1)[0]
    pool.recurrent_states[:, req.linear_slot_idx].fill_(3.0)
    cm.spec_snapshot(req)
    pool.recurrent_states[:, req.linear_slot_idx].fill_(-1.0)

    verdict = SpecVerdict(accepted=[70, 71, 99], n_accepted=2, bonus=True,
                          committed=10, replay_from=7, replay_to=7)
    finalize_spec_step(req, verdict, cm)
    # all accepted: the advanced state already encodes the committed prefix, no restore
    assert (pool.recurrent_states[:, req.linear_slot_idx] == -1.0).all()
    assert (req.cached_len, req.device_len) == (10, 11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
def test_spec_probs_runs_on_live_forward_logits():
    """Device parity: verify runs on GPU forward logits (top-k/top-p masks must live
    with them). Live RuntimeError signature: keep (cpu) &= probs>=kth (cuda:0)."""
    from freetoken.engine.spec import spec_probs as _sp

    gen = torch.Generator(device="cuda").manual_seed(0)
    logits = torch.randn(32, generator=gen, device="cuda")
    out = _sp(logits, _params(top_k=5, top_p=0.9))
    assert out.device.type == "cuda"
    assert float(out.sum().item()) == pytest.approx(1.0)


def test_verify_and_finalize_rewinds_the_post_forward_frame():
    """Live-contract regression: forward_batch's complete_n advances lengths BEFORE the
    post-forward verify runs, so finalize must see the pre-forward reservation frame.
    Without the rewind, accept_spec_tail asserts (committed < cached) and the page
    release overshoots the reservation end (live AssertionError: (2, 67, 68))."""
    from freetoken.engine.mtp import SpecAccounting, verify_and_finalize
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.table import TableManager

    page_size, num_pages = 1, 64
    tm = TableManager(2, torch.zeros((2, num_pages * page_size), dtype=torch.int32))
    cm = CacheManager(num_pages, page_size, tm.page_table, "radix")
    req = Req(
        input_ids=torch.arange(11, dtype=torch.int64),
        table_idx=tm.allocate(),
        cached_len=10, output_len=16, uid=7,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
        cache_handle=None,
    )
    req.reserve_spec(2)  # pre-forward frame: cached 10, device 13
    cm.allocate_paged([req])
    req.complete_n(3)  # the scoring forward's advance: cached 13, device 14
    V4, d = 4, 2
    one_hot = lambda i: torch.nn.functional.one_hot(torch.tensor(i), V4).float()
    draft_tokens = torch.tensor([1, 2])
    draft_logits = torch.stack([one_hot(1), one_hot(2)])
    target_logits = torch.stack([one_hot(1), one_hot(2), one_hot(3)])
    verify_and_finalize(
        req, draft_tokens=draft_tokens, draft_logits=draft_logits,
        target_logits=target_logits, base_len=10, cache_manager=cm,
        accounting=SpecAccounting())
    assert req.input_ids.tolist() == list(range(11)) + [1, 2, 3]
    assert (req.cached_len, req.device_len, req.spec_depth) == (13, 14, 0)


def test_greedy_fast_path_covers_every_mismatch_position():
    """The id-compare fast path must equal the general rule at every rejection
    position: reject at 0, at the last, and all-match (bonus)."""
    base = 30
    greedy = _params(temperature=0.0)
    # target argmax rows: 4, 4, 3 (bonus row 3 -> argmax 5)
    p = torch.stack([_onehot(4), _onehot(4), _onehot(3), _onehot(5)])
    q = torch.stack([_onehot(1), _onehot(2), _onehot(3)])

    v = verify_step(torch.tensor([1, 2, 3]), q, p, greedy, base)
    assert (v.accepted, v.n_accepted, v.bonus) == ([4], 0, False)
    assert (v.committed, v.replay_from, v.replay_to) == (base + 1, base, base + 1)

    v = verify_step(torch.tensor([4, 2, 3]), q, p, greedy, base)
    assert (v.accepted, v.n_accepted, v.bonus) == ([4, 4], 1, False)
    assert v.committed == base + 2

    v = verify_step(torch.tensor([4, 4, 3]), q, p, greedy, base)
    assert (v.accepted, v.n_accepted, v.bonus) == ([4, 4, 3, 5], 3, True)
    assert not v.needs_replay


def test_greedy_fast_path_ignores_the_draft_distribution():
    """Acceptance depends only on the target argmax match, mirroring the general rule
    (q[x] == 0 still accepts when p[x] == 1)."""
    greedy = _params(temperature=0.0)
    p = torch.stack([_onehot(4), _onehot(0)])
    q = torch.stack([_onehot(2)])  # the draft token 4 is outside q's support
    v = verify_step(torch.tensor([4]), q, p, greedy, base_len=5)
    # accepted despite q[4] == 0; the bonus follows the second target row
    assert (v.accepted, v.bonus) == ([4, 0], True)


def test_greedy_fast_path_empty_drafts_is_bonus_only():
    greedy = _params(top_k=1)  # top_k == 1 is greedy too
    p = torch.stack([_onehot(5)])
    v = verify_step(
        torch.empty(0, dtype=torch.int64), torch.empty(0, V), p, greedy, base_len=5)
    assert (v.accepted, v.n_accepted, v.bonus) == ([5], 0, True)
    assert not v.needs_replay


def test_greedy_verify_accepts_a_missing_draft_distribution():
    """A deterministic proposal (prompt look-up) carries no draft distribution; the
    greedy fast path compares ids only, so None is valid there."""
    greedy = _params(temperature=0.0)
    p = torch.stack([_onehot(3), _onehot(5)])
    verdict = verify_step(torch.tensor([3]), None, p, greedy, base_len=10)
    assert verdict.accepted == [3, 5] and verdict.bonus
    mismatch = verify_step(torch.tensor([1]), None, p, greedy, base_len=10)
    assert mismatch.accepted == [3] and not mismatch.bonus
