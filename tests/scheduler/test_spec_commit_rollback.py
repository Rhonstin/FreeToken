"""Speculative KV pages and GDN state commit/rollback.

The state side of a speculative step: the forward advances the live GDN slot in place
through every draft row, so a rejected suffix must roll back to the pre-step snapshot;
the rejected KV reservation tail goes back to the page pool; the per-row metadata must
span rows, not requests; and admission must count the reserved draft rows as in flight.
Plain decode never touches any of this, so every test pins both sides.
"""

from __future__ import annotations

import torch

import pytest

from freetoken.core import Req, SamplingParams
from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.table import TableManager


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _req(uid: int, prompt: int, output: int, cached: int) -> Req:
    """A request whose prefill already ran (cached < prompt is the admission invariant)."""
    return Req(
        input_ids=torch.arange(prompt, dtype=torch.int64),
        table_idx=uid,
        cached_len=cached,
        output_len=output,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=output),
        cache_handle=None,
    )


def _at_frame(req: Req, cached: int) -> Req:
    """Post-forward frame: the request is authoritative through ``cached`` tokens."""
    req.cached_len = cached
    req.device_len = cached + 1
    req.spec_depth = 0
    return req


def _hybrid_manager(page_size: int = 4, num_pages: int = 32, num_slots: int = 8):
    """A hybrid CacheManager with a real CPU LinearStatePool (no engine needed)."""
    group = LinearGatedDeltaGroupConfig(
        name="linear",
        layer_ids=(0,),
        num_key_heads=1,
        num_value_heads=1,
        key_head_dim=8,
        value_head_dim=8,
        conv_kernel_dim=4,
        output_gate="sigmoid",
    )
    device = torch.device("cpu")
    pool = LinearStatePool(group, num_slots, torch.float32, device)
    table = torch.zeros((4, num_pages * page_size), dtype=torch.int32, device=device)
    return CacheManager(
        num_pages, page_size, table, "hybrid_radix", linear_state_pool=pool,
    ), pool


def _radix_manager(page_size: int = 4, num_pages: int = 8):
    table_manager = TableManager(
        2, torch.zeros((2, num_pages * page_size), dtype=torch.int32))
    return (
        CacheManager(num_pages, page_size, table_manager.page_table, "radix"),
        table_manager,
    )


def _live_state(pool: LinearStatePool, slot: int) -> torch.Tensor:
    """A clone of everything spec_snapshot journals (recurrent + conv + slot_states)."""
    return torch.cat([
        pool.recurrent_states[:, slot].flatten(),
        pool.conv_states[:, slot].flatten(),
    ])


# ======================================================================================
# Journal: snapshot before the forward, restore on rejection
# ======================================================================================


def test_spec_snapshot_journals_the_live_slot_and_restores_it():
    cm, pool = _hybrid_manager()
    req = _req(1, prompt=12, output=16, cached=11)
    _at_frame(req, 11)
    req.table_idx = 0
    req.linear_slot_idx = pool.alloc(1)[0]
    pool.recurrent_states[:, req.linear_slot_idx].fill_(3.0)
    pool.conv_states[:, req.linear_slot_idx].fill_(0.5)

    req.reserve_spec(2)
    cm.spec_snapshot(req)
    assert req.spec_journal_slot is not None
    assert req.spec_journal_slot != req.linear_slot_idx
    journal = req.spec_journal_slot

    # the forward advances the live slot in place through every draft row
    pool.recurrent_states[:, req.linear_slot_idx].fill_(-1.0)
    pool.conv_states[:, req.linear_slot_idx].fill_(-2.0)

    cm.spec_restore(req)
    assert torch.equal(_live_state(pool, req.linear_slot_idx), _live_state(pool, journal))
    assert (pool.recurrent_states[:, req.linear_slot_idx] == 3.0).all()
    assert (pool.conv_states[:, req.linear_slot_idx] == 0.5).all()


def test_spec_snapshot_reuses_the_journal_slot_across_steps():
    cm, pool = _hybrid_manager()
    req = _req(1, prompt=12, output=16, cached=11)
    _at_frame(req, 11)
    req.table_idx = 0
    req.linear_slot_idx = pool.alloc(1)[0]
    free_after_alloc = pool.num_free_slots

    req.reserve_spec(2)
    cm.spec_snapshot(req)
    first = req.spec_journal_slot
    assert pool.num_free_slots == free_after_alloc - 1

    # the next step re-snapshots into the same slot: no further allocation
    req.reserve_spec(1)
    cm.spec_snapshot(req)
    assert req.spec_journal_slot == first
    assert pool.num_free_slots == free_after_alloc - 1


def test_spec_snapshot_and_restore_are_noops_without_hybrid_state():
    cm, _tm = _radix_manager()
    req = _req(1, prompt=12, output=16, cached=11)
    _at_frame(req, 11)
    req.reserve_spec(2)
    cm.spec_snapshot(req)  # must not allocate, must not raise
    assert req.spec_journal_slot is None
    cm.spec_restore(req)  # noqa: no journal, no live slot -- still a no-op


def test_spec_snapshot_skips_a_plain_step():
    cm, pool = _hybrid_manager()
    req = _req(1, prompt=12, output=16, cached=11)
    _at_frame(req, 11)
    req.table_idx = 0
    req.linear_slot_idx = pool.alloc(1)[0]
    free_before = pool.num_free_slots
    cm.spec_snapshot(req)  # spec_depth == 0: nothing to roll back to
    assert req.spec_journal_slot is None
    assert pool.num_free_slots == free_before


def test_finish_returns_the_journal_slot_and_is_idempotent():
    cm, pool = _hybrid_manager()
    req = _req(1, prompt=12, output=16, cached=11)
    _at_frame(req, 11)
    req.table_idx = 0
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    free_before = pool.num_free_slots

    req.reserve_spec(2)
    cm.spec_snapshot(req)
    assert pool.num_free_slots == free_before - 1

    cm._free_req_slots(req)
    assert pool.num_free_slots == free_before + 3  # live + 2 ping-pong + journal
    assert req.spec_journal_slot is None and req.linear_slot_idx is None
    cm._free_req_slots(req)  # second free releases nothing
    assert pool.num_free_slots == free_before + 3


# ======================================================================================
# Commit: one call pairs the tail release with the length rotation
# ======================================================================================


def test_commit_spec_releases_the_tail_and_rotates_lengths():
    page_size = 4
    cm, table_manager = _radix_manager(page_size=page_size, num_pages=8)
    req = _req(1, prompt=8, output=16, cached=7)
    _at_frame(req, 8)
    req.table_idx = table_manager.allocate()
    req.reserve_spec(5)  # device 14: alloc fills pages 2 and 3
    cm.allocate_paged([req])
    snapshot = table_manager.page_table[req.table_idx].clone()
    free_before = len(cm.free_slots)

    # accepted through 10: page 2 holds committed tokens 8..10 -> kept; page 3 released
    cm.commit_spec(req, committed=10)
    assert len(cm.free_slots) == free_before + 1
    assert (req.cached_len, req.device_len, req.spec_depth) == (10, 11, 0)
    assert torch.equal(table_manager.page_table[req.table_idx, :10], snapshot[:10])


def test_commit_spec_full_acceptance_releases_nothing():
    page_size = 4
    cm, table_manager = _radix_manager(page_size=page_size, num_pages=8)
    req = _req(1, prompt=8, output=16, cached=7)
    _at_frame(req, 8)
    req.table_idx = table_manager.allocate()
    req.reserve_spec(3)  # device 12: alloc fills exactly page 2
    cm.allocate_paged([req])
    free_before = len(cm.free_slots)

    cm.commit_spec(req, committed=12)
    assert len(cm.free_slots) == free_before
    assert (req.cached_len, req.device_len, req.spec_depth) == (12, 13, 0)


# ======================================================================================
# Per-row GDN metadata spans rows, not requests
# ======================================================================================


def test_gdn_row_slots_expand_per_spec_row():
    from freetoken.scheduler.scheduler import _gdn_row_slots

    req = _req(1, prompt=20, output=12, cached=13)
    _at_frame(req, 13)
    req.linear_slot_idx = 5
    req.reserve_spec(2)

    class FakeBatch:
        spec_active = True
        spec_rows = [(req, 0), (req, 1), (req, 2)]
        padded_reqs = [req]

    assert _gdn_row_slots(FakeBatch(), padding_slot=0) == [5, 5, 5]


def test_gdn_row_slots_plain_batch_is_one_slot_per_request():
    from freetoken.scheduler.scheduler import _gdn_row_slots

    reqs = [_req(uid, prompt=11, output=8, cached=10) for uid in (1, 2)]
    for req in reqs:
        _at_frame(req, 10)
    reqs[0].linear_slot_idx = 3  # reqs[1] keeps None -> padding slot

    class FakeBatch:
        spec_active = False
        spec_rows = None
        padded_reqs = reqs

    assert _gdn_row_slots(FakeBatch(), padding_slot=0) == [3, 0]


def test_fla_metadata_spec_decode_frames_each_span_as_one_sequence():
    """A spec request's anchor + draft rows run as ONE continuation sequence (single
    live slot): parallel same-slot rows race, corrupting the per-row outputs the
    verification compares against and the written state."""
    from freetoken.attention.linear import build_fla_metadata
    from types import SimpleNamespace

    req = _req(1, prompt=20, output=12, cached=13)
    _at_frame(req, 13)
    req.linear_slot_idx = 5
    req.reserve_spec(2)
    batch = SimpleNamespace(
        is_decode=True,
        spec_active=True,
        spec_rows=[(req, 0), (req, 1), (req, 2)],
        padded_reqs=[req],
        linear_table_idx=torch.tensor([5, 5, 5], dtype=torch.int32),
    )
    meta = build_fla_metadata(batch, torch.device("cpu"))
    assert meta.cu_seqlens.tolist() == [0, 3]
    assert meta.cache_indices.tolist() == [5]
    assert meta.has_initial_state.tolist() == [True]
    assert meta.fresh_state_indices is None


def test_fla_metadata_spec_decode_groups_concurrent_requests():
    from freetoken.attention.linear import build_fla_metadata
    from types import SimpleNamespace

    r1 = _req(1, prompt=20, output=12, cached=13)
    r2 = _req(2, prompt=8, output=8, cached=6)
    for req, slot in ((r1, 3), (r2, 7)):
        _at_frame(req, req.cached_len)
        req.linear_slot_idx = slot
    r1.reserve_spec(2)
    r2.reserve_spec(1)
    batch = SimpleNamespace(
        is_decode=True,
        spec_active=True,
        spec_rows=[(r1, 0), (r1, 1), (r1, 2), (r2, 0), (r2, 1)],
        padded_reqs=[r1, r2],
        linear_table_idx=torch.tensor([3, 3, 3, 7, 7], dtype=torch.int32),
    )
    meta = build_fla_metadata(batch, torch.device("cpu"))
    assert meta.cu_seqlens.tolist() == [0, 3, 5]
    assert meta.cache_indices.tolist() == [3, 7]
    assert meta.has_initial_state.tolist() == [True, True]


def test_fla_metadata_plain_decode_is_unchanged():
    from freetoken.attention.linear import build_fla_metadata
    from types import SimpleNamespace

    reqs = [_req(uid, prompt=11, output=8, cached=10) for uid in (1, 2)]
    for req in reqs:
        _at_frame(req, 10)
    slots = torch.tensor([1, 2], dtype=torch.int32)
    batch = SimpleNamespace(
        is_decode=True,
        spec_active=False,
        spec_rows=None,
        padded_reqs=reqs,
        linear_table_idx=slots,
    )
    meta = build_fla_metadata(batch, torch.device("cpu"))
    assert meta.cu_seqlens.tolist() == [0, 1, 2]
    assert torch.equal(meta.cache_indices, slots)


# ======================================================================================
# Admission accounting counts the reserved draft rows
# ======================================================================================


def test_inflight_tokens_plain_decode_is_unchanged():
    dm = DecodeManager(page_size=4)
    req = _req(1, prompt=11, output=8, cached=10)
    _at_frame(req, 10)
    dm.running_reqs = {req}
    # the old formula exactly: remain + one page of slack
    assert dm.inflight_tokens == req.remain_len + 3


def test_inflight_tokens_counts_reserved_draft_rows():
    dm = DecodeManager(page_size=4)
    plain = _req(1, prompt=11, output=8, cached=10)
    _at_frame(plain, 10)
    spec = _req(2, prompt=20, output=12, cached=13)
    _at_frame(spec, 13)
    spec.reserve_spec(2)
    dm.running_reqs = {plain, spec}
    assert dm.inflight_tokens == (
        plain.remain_len + spec.remain_len + 2 * 3 + 2  # +2 reserved draft rows
    )


# ======================================================================================
# Verify write-back + GDN continuation replay (the 419.12 seam)
# ======================================================================================

from types import SimpleNamespace

from freetoken.core import Batch
from freetoken.scheduler.scheduler import ForwardInput, Scheduler


def _replay_req():
    req = _req(1, prompt=20, output=12, cached=10)
    _at_frame(req, 16)  # post-commit frame: cached 16, device 17
    req.table_idx = 0
    return req


def _stub_engine(page_table, calls):
    return SimpleNamespace(
        page_table=page_table,
        linear_state_pool=None,  # replay skips fla metadata without GDN tiers
        attn_backend=SimpleNamespace(
            prepare_metadata=lambda batch: calls.append(("meta", batch.phase))),
        sampler=SimpleNamespace(prepare=lambda batch: calls.append(("sample", None))),
        device=torch.device("cpu"),
        forward_batch=lambda batch, args: calls.append(("forward", batch)) or None,
    )


def test_verify_writes_back_accepted_tokens_over_sampled_rows(monkeypatch):
    """Under stochastic sampling the forward's rows differ from the verdict: the pool
    must hold verified tokens (replay inputs + next anchor read them)."""
    import freetoken.scheduler.scheduler as sched_mod

    pool = torch.zeros((2, 32), dtype=torch.int32)
    pool[0, 11:14] = torch.tensor([70, 71, 72])  # the forward's sampled rows
    verdict = SimpleNamespace(accepted=[7, 8, 9], committed=13, needs_replay=False)
    monkeypatch.setattr(
        sched_mod, "verify_and_finalize", lambda *a, **k: verdict)
    sched = SimpleNamespace(
        token_pool=pool,
        cache_manager=SimpleNamespace(is_hybrid=False),
        decode_manager=SimpleNamespace(spec_depth_fn=None),
        spec_accounting=None,
    )
    req = _replay_req()
    req.spec_depth = 2
    batch = Batch(reqs=[req], phase="decode")
    batch.spec_rows = [(req, 0), (req, 1), (req, 2)]
    out = SimpleNamespace(spec_logits=torch.zeros(3, 5))
    Scheduler._verify_spec_batch(sched, batch, out, {1: (10, None, None)})
    assert pool[0, 11:14].tolist() == [7, 8, 9]
    assert batch.spec_accepted == {1: [7, 8, 9]}
    assert batch.spec_finalized is True


def test_verify_replays_partial_accept_on_hybrid_only(monkeypatch):
    import freetoken.scheduler.scheduler as sched_mod

    verdict = SimpleNamespace(accepted=[7], committed=11, needs_replay=True)
    monkeypatch.setattr(
        sched_mod, "verify_and_finalize", lambda *a, **k: verdict)
    pool = torch.zeros((2, 32), dtype=torch.int32)
    replayed = []
    sched = SimpleNamespace(
        token_pool=pool,
        cache_manager=SimpleNamespace(is_hybrid=True),
        decode_manager=SimpleNamespace(spec_depth_fn=None),
        spec_accounting=None,
        _replay_spec_suffix=lambda req, v: replayed.append((req.uid, v)),
    )
    req = _replay_req()
    req.spec_depth = 1
    batch = Batch(reqs=[req], phase="decode")
    batch.spec_rows = [(req, 0), (req, 1)]
    out = SimpleNamespace(spec_logits=torch.zeros(2, 5))
    Scheduler._verify_spec_batch(sched, batch, out, {1: (10, None, None)})
    assert replayed == [(1, verdict)]


def test_replay_suffix_frames_a_continuation_chunk_without_allocating():
    page_table = torch.arange(64, dtype=torch.int32).view(2, 32)
    pool = torch.arange(64, dtype=torch.int32).view(2, 32)
    calls = []
    sched = SimpleNamespace(
        device=torch.device("cpu"),
        token_pool=pool,
        engine=_stub_engine(page_table, calls),
    )
    req = _replay_req()  # cached 16, device 17
    verdict = SimpleNamespace(replay_from=10, replay_to=14)
    Scheduler._replay_spec_suffix(sched, req, verdict)
    kinds = [k for k, _ in calls]
    assert kinds == ["meta", "sample", "forward"]
    fwd_batch = calls[-1][1]
    assert fwd_batch.phase == "prefill"
    assert fwd_batch.input_ids.tolist() == pool[0, 10:14].tolist()
    assert fwd_batch.positions.tolist() == [10, 11, 12, 13]
    assert torch.equal(fwd_batch.out_loc, page_table[0, 10:14])
    # lengths restored: the replay refills state, it does not advance the request
    assert (req.cached_len, req.device_len) == (16, 17)


def test_forward_skips_pool_write_for_finalized_batches():
    """The forward's sampled rows must not clobber the verify pass's write-back."""
    pool = torch.full((2, 32), -1, dtype=torch.int32)
    written = torch.tensor([5], dtype=torch.int32)
    out_map = (torch.tensor([0]), torch.tensor([20]))
    engine = SimpleNamespace(
        forward_batch=lambda batch, args: SimpleNamespace(next_tokens_gpu=written),
    )
    sched = SimpleNamespace(
        _mtp_enabled=False,
        token_pool=pool,
        toolcall_anchor_id=None,
        cache_manager=SimpleNamespace(
            spec_snapshot=lambda req: None),
        engine=engine,
        decode_manager=SimpleNamespace(
            filter_reqs=lambda reqs: setattr(sched, "filtered", True)),
    )
    req = _replay_req()
    batch = Batch(reqs=[req], phase="decode")
    batch.spec_rows = [(req, 0)]
    batch.spec_finalized = True  # verified earlier; pool already holds verdict tokens
    pool[0, 20] = 42
    Scheduler._forward(
        sched, ForwardInput(batch=batch, sample_args=None,
                            input_tuple=(None, None), write_tuple=out_map))
    assert pool[0, 20].item() == 42  # untouched
    assert sched.filtered is True
