"""Multi-token (speculative / MTP) decode scheduling.

The scheduler generalization under test: a decode step carries responses for one anchor
plus up to ``spec_depth`` draft rows per request, KV pages are reserved for rows that
verification may reject, and per-request advancement is variable. Plain decode must stay
byte-for-byte the old path, so every test pins both sides.
"""

from __future__ import annotations

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.table import TableManager


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


# ======================================================================================
# Req frame primitives
# ======================================================================================


def test_reserve_spec_establishes_the_frame_idempotently():
    req = _req(1, prompt=11, output=8, cached=10)
    _at_frame(req, 14)
    req.reserve_spec(3)
    assert req.device_len == 18  # cached_len + 1 anchor + 3 draft rows
    assert req.spec_depth == 3
    # re-establishing (a re-schedule, or a stale un-drained frame) overwrites, never grows
    req.reserve_spec(1)
    assert req.device_len == 16 and req.spec_depth == 1
    req.reserve_spec(0)
    assert req.device_len == 15 and req.spec_depth == 0


def test_reserve_spec_rejects_a_frame_past_the_budget():
    req = _req(1, prompt=10, output=4, cached=9)  # max_device_len = 14
    try:
        req.reserve_spec(6)  # cached 9 + 1 + 6 = 16 > 14
    except AssertionError:  # noqa: PT011
        pass
    else:
        raise AssertionError("device_len would exceed max_device_len")


def test_commit_spec_releases_the_reservation_and_rotates_the_frame():
    req = _req(1, prompt=11, output=16, cached=10)
    _at_frame(req, 12)
    req.reserve_spec(3)  # device 16
    # verification accepted through 15 (anchor + 2 drafts): the 3rd draft is dropped
    req.commit_spec(15)
    assert (req.cached_len, req.device_len, req.spec_depth) == (15, 16, 0)
    # the next step re-establishes from cached_len
    req.reserve_spec(2)
    assert req.device_len == 18


def test_release_spec_reservation_drops_back_to_the_plain_frame():
    req = _req(1, prompt=11, output=8, cached=10)
    _at_frame(req, 12)
    req.reserve_spec(4)  # device 17
    req.release_spec_reservation()
    assert (req.cached_len, req.device_len, req.spec_depth) == (12, 13, 0)


def test_complete_one_and_n_agree_for_plain_decode():
    req = _req(1, prompt=11, output=8, cached=10)
    _at_frame(req, 10)
    req.complete_n(1)
    assert (req.cached_len, req.device_len) == (11, 12)
    req.complete_one()
    assert (req.cached_len, req.device_len) == (12, 13)


def test_complete_n_advances_a_multi_row_step():
    req = _req(1, prompt=11, output=16, cached=10)
    _at_frame(req, 10)
    req.reserve_spec(3)
    n_rows = req.spec_depth + 1
    assert n_rows == 4
    req.complete_n(n_rows)
    assert req.cached_len == 14 and req.device_len == 15


def test_complete_n_of_extend_len_matches_complete_one_on_a_prefill_chunk():
    """complete_n must take the rows actually forwarded: a chunked prefill frame with
    extend > 1 advances exactly like the old complete_one (cached = device), never by 1."""
    req = _req(1, prompt=11, output=8, cached=10)
    req.cached_len = 0  # a fresh prefill chunk: 11 rows in flight
    assert req.extend_len == 11
    req.complete_n(req.extend_len)
    assert (req.cached_len, req.device_len) == (11, 12)


# ======================================================================================
# DecodeManager: batch rows + frame re-establishment
# ======================================================================================


def test_plain_schedule_unchanged_without_a_depth_fn():
    dm = DecodeManager(page_size=4)
    req = _req(1, prompt=11, output=8, cached=10)
    _at_frame(req, 10)
    dm.running_reqs = {req}
    batch = dm.schedule_next_batch()
    assert batch.spec_rows is None
    assert req.spec_depth == 0
    assert req.cached_len == 10 and req.device_len == 11  # untouched: one row at [10, 11)


def test_spec_schedule_establishes_frames_and_rows():
    dm = DecodeManager(page_size=4)
    dm.spec_depth_fn = lambda req: 2
    reqs = [_req(uid, prompt=20, output=12, cached=13) for uid in (2, 1)]
    for req in reqs:
        _at_frame(req, 13)
    dm.running_reqs = set(reqs)
    batch = dm.schedule_next_batch()
    assert batch.spec_active
    # uid order: req 1 first; each request expands to anchor + 2 drafts
    assert batch.spec_rows[0][0].uid == 1
    assert [spec_index for _req, spec_index in batch.spec_rows] == [0, 1, 2] * 2
    for req in reqs:
        assert req.spec_depth == 2
        assert (req.device_len - req.cached_len) == 3
    assert len(batch.spec_rows) == 6


def test_spec_schedule_clamps_depth_to_the_output_budget():
    dm = DecodeManager(page_size=4)
    dm.spec_depth_fn = lambda req: 8
    # one slot past the frame stays free for the committed token: room = max - cached - 2
    req = _req(1, prompt=16, output=1, cached=14)  # room = max(17) - 14 - 2 = 1
    _at_frame(req, 14)
    dm.running_reqs = {req}
    batch = dm.schedule_next_batch()
    assert req.spec_depth == 1  # clamped, not dropped
    assert (req.device_len - req.cached_len) == 2
    assert batch.spec_active
    # full accept commits at max - 1, never past the id buffer (live assert (71, 71))
    committed = req.cached_len + req.spec_depth + 1
    assert committed == req.max_device_len - 1


def test_spec_schedule_clamps_to_zero_keeps_the_plain_frame():
    dm = DecodeManager(page_size=4)
    dm.spec_depth_fn = lambda req: 8
    req = _req(1, prompt=16, output=1, cached=14)
    _at_frame(req, 15)  # room = max(17) - 15 - 2 = 0
    dm.running_reqs = {req}
    batch = dm.schedule_next_batch()
    assert req.spec_depth == 0
    assert req.device_len == req.cached_len + 1
    assert not batch.spec_active


# ======================================================================================
# Scheduler mappings: rows -> positions / token-pool write slots
# ======================================================================================


def _batch_from_manager(reqs, spec):
    dm = DecodeManager(page_size=4)
    if spec:
        dm.spec_depth_fn = lambda req: 2  # noqa: E731
    dm.running_reqs = set(reqs)
    return dm.schedule_next_batch()


def test_positions_and_write_map_one_row_per_plain_req():
    from freetoken.scheduler.scheduler import _make_positions, _make_write_tuple

    req = _req(1, prompt=11, output=8, cached=10)
    _at_frame(req, 10)
    batch = _batch_from_manager([req], spec=False)
    batch.padded_reqs = batch.reqs
    device = torch.device("cpu")
    assert _make_positions(batch, device).tolist() == [10]
    write_mapping, write_position = _make_write_tuple(batch, device)
    assert write_mapping.tolist() == [req.table_idx]
    assert write_position.tolist() == [11]


def test_positions_and_write_map_expand_per_spec_row():
    from freetoken.scheduler.scheduler import _make_positions, _make_write_tuple

    req = _req(1, prompt=20, output=12, cached=13)
    _at_frame(req, 13)
    batch = _batch_from_manager([req], spec=True)
    batch.padded_reqs = batch.reqs
    device = torch.device("cpu")
    assert (req.device_len - req.cached_len) == 3  # 1 anchor + 2 drafts
    assert _make_positions(batch, device).tolist() == [13, 14, 15]
    write_mapping, write_position = _make_write_tuple(batch, device)
    assert write_mapping.tolist() == [req.table_idx] * 3
    # one continuation slot per row: 14, 15, 16 -- 16 is one past the reserved frame,
    # exactly how a plain decode writes its next-step anchor
    assert write_position.tolist() == [14, 15, 16]


# ======================================================================================
# CacheManager: page-granular release of rejected reservations
# ======================================================================================


def test_free_token_tail_releases_whole_pages_only():
    page_size = 4
    num_reqs, num_pages = 2, 8
    table_manager = TableManager(num_reqs, torch.zeros((num_reqs, num_pages * page_size), dtype=torch.int32))
    cm = CacheManager(num_pages, page_size, table_manager.page_table, "radix")
    req = _req(1, prompt=8, output=16, cached=7)
    _at_frame(req, 8)
    req.table_idx = table_manager.allocate()
    req.reserve_spec(9)  # device 18: alloc fills pages 2,3,4 -> token slots for pos 8..19
    cm.allocate_paged([req])
    snapshot = table_manager.page_table[req.table_idx].clone()
    free_before = len(cm.free_slots)
    # accepted through 12: pages wholly beyond 12 (pages 3 and 4) are the tail
    cm.free_token_tail(req, keep_len=12)
    assert len(cm.free_slots) == free_before + 2  # 2 page entries returned
    # the committed region is untouched
    assert torch.equal(table_manager.page_table[req.table_idx, :12], snapshot[:12])


def test_free_token_tail_does_not_touch_a_page_the_prefix_shares():
    page_size = 4
    num_reqs, num_pages = 2, 8
    table_manager = TableManager(num_reqs, torch.zeros((num_reqs, num_pages * page_size), dtype=torch.int32))
    cm = CacheManager(num_pages, page_size, table_manager.page_table, "radix")
    req = _req(1, prompt=8, output=16, cached=7)
    _at_frame(req, 8)
    req.table_idx = table_manager.allocate()
    req.reserve_spec(9)  # device 18
    cm.allocate_paged([req])
    free_before = len(cm.free_slots)
    # accept through 13: page 3 holds the committed token 12 -> untouched; page 4 released
    cm.free_token_tail(req, keep_len=13)
    assert len(cm.free_slots) == free_before + 1  # only page 4 (positions 16..19)


def test_free_token_tail_is_a_noop_without_a_tail():
    page_size = 1
    num_reqs, num_pages = 2, 32
    table_manager = TableManager(num_reqs, torch.zeros((num_reqs, num_pages), dtype=torch.int32))
    cm = CacheManager(num_pages, page_size, table_manager.page_table, "radix")
    req = _req(1, prompt=8, output=24, cached=7)
    _at_frame(req, 8)
    req.table_idx = table_manager.allocate()
    cm.allocate_paged([req])
    req.reserve_spec(7)  # device 16
    cm.allocate_paged([req])
    free_before = len(cm.free_slots)
    assert len(cm.free_slots) == free_before
    cm.free_token_tail(req, keep_len=req.device_len - 4)
    # page_size 1: every slot beyond 12 returns individually
    assert len(cm.free_slots) == free_before + 4


# ======================================================================================
# Draft closure: the retained anchor residual reaches draft proposal (seam 1)
# ======================================================================================


def _draft_sched(page_table: torch.Tensor, anchor: torch.Tensor, uid: int = 1):
    """A Scheduler double with a retained anchor residual and stub engine pieces."""
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    mtp_head = SimpleNamespace(draft_layer_ids=lambda: [48], last_draft_residual=None)
    engine = SimpleNamespace(
        model=SimpleNamespace(mtp=mtp_head),
        page_table=page_table,
        attn_backend=SimpleNamespace(prepare_metadata=lambda batch: None),
    )
    sched = Scheduler.__new__(Scheduler)
    sched.engine = engine
    sched.device = torch.device("cpu")
    sched.token_pool = torch.zeros((2, 64), dtype=torch.int32)
    # anchor spans: tests frame reqs at cached_len 10, so base 9 exposes row 0
    span = anchor.unsqueeze(0) if anchor.dim() == 1 else anchor
    sched._last_target_hidden = {uid: (9, span)}
    return sched, mtp_head


def test_draft_step_fn_returns_a_callable_proposal():
    sched, _head = _draft_sched(
        torch.zeros(2, 64, dtype=torch.int32), torch.zeros(4 * 8))
    req = _req(1, prompt=11, output=8, cached=10)
    _at_frame(req, 10)

    from freetoken.scheduler.scheduler import Scheduler

    assert callable(Scheduler._draft_step_fn(sched, req))


def test_draft_step_fn_chains_anchor_then_draft_residuals(monkeypatch):
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import Scheduler

    seen: dict = {}

    def fake_draft_forward(model, ids, rows, batch):
        seen["ids"] = ids
        seen["rows"] = rows
        seen["batch"] = batch
        return torch.zeros(ids.numel(), 5)

    monkeypatch.setattr(sched_mod, "draft_forward", fake_draft_forward)
    anchor = torch.arange(32, dtype=torch.float32)
    page_table = torch.arange(128, dtype=torch.int32).view(2, 64)
    sched, head = _draft_sched(page_table, anchor)
    req = _req(1, prompt=11, output=8, cached=10)
    _at_frame(req, 10)
    req.table_idx = 0
    step_fn = Scheduler._draft_step_fn(sched, req)

    out = step_fn(torch.tensor([7]))
    assert out.shape == (1, 5)
    assert torch.equal(seen["rows"], anchor.unsqueeze(0))
    batch = seen["batch"]
    assert batch.phase == "prefill" and batch.spec_rows is None
    assert batch.positions.tolist() == [10]
    assert torch.equal(batch.out_loc, page_table[0, 10:11])
    assert torch.equal(batch.input_ids, torch.tensor([7]))
    assert seen["ids"].dtype == torch.int32  # token-pool width, like the gather

    head.last_draft_residual = torch.full((1, 32), 2.0)
    out = step_fn(torch.tensor([7, 9]))
    assert out.shape == (2, 5)
    assert torch.equal(seen["rows"][0], anchor)
    assert torch.equal(seen["rows"][1], torch.full((32,), 2.0))
    assert batch is not seen["batch"]  # the batch reframes per prefix length
    assert seen["batch"].positions.tolist() == [10, 11]


def test_draft_step_fn_picks_the_row_before_the_anchor(monkeypatch):
    """(h_{p-1}, x_p): the anchor's hidden is the row BEFORE its position -- the
    last accepted row -- not the anchor input row (live acceptance was degraded
    by exactly this off-by-one)."""
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import Scheduler

    seen: dict = {}

    def fake_draft_forward(model, ids, rows, batch):
        seen["rows"] = rows
        return torch.zeros(ids.numel(), 5)

    monkeypatch.setattr(sched_mod, "draft_forward", fake_draft_forward)
    span = torch.stack([torch.full((32,), float(i)) for i in range(3)])
    sched, _head = _draft_sched(torch.zeros(2, 64, dtype=torch.int32), span)
    sched._last_target_hidden = {1: (20, span)}
    req = _req(1, prompt=30, output=8, cached=23)  # committed = base + 3 rows
    req.table_idx = 0
    step_fn = Scheduler._draft_step_fn(sched, req)
    step_fn(torch.tensor([7]))
    assert torch.equal(seen["rows"], span[2:3])


def test_draft_step_fn_rejects_a_broken_hidden_chain(monkeypatch):
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import Scheduler

    monkeypatch.setattr(
        sched_mod, "draft_forward", lambda *a: torch.zeros(2, 5))
    sched, _head = _draft_sched(
        torch.zeros(2, 64, dtype=torch.int32), torch.zeros(32))
    req = _req(1, prompt=11, output=8, cached=10)
    _at_frame(req, 10)
    step_fn = Scheduler._draft_step_fn(sched, req)
    import pytest

    with pytest.raises(RuntimeError, match="hidden chain broke"):
        step_fn(torch.tensor([7, 9]))


def test_draft_step_fn_names_a_missing_anchor_or_head():
    import pytest
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import Scheduler

    sched, _head = _draft_sched(torch.zeros(2, 64, dtype=torch.int32), torch.zeros(32))
    other = _req(2, prompt=11, output=8, cached=10)
    _at_frame(other, 10)
    with pytest.raises(RuntimeError, match="no retained anchor hidden"):
        Scheduler._draft_step_fn(sched, other)
    sched.engine = SimpleNamespace(model=SimpleNamespace(mtp=None))
    req = _req(1, prompt=11, output=8, cached=10)
    with pytest.raises(RuntimeError, match="built no MTP draft head"):
        Scheduler._draft_step_fn(sched, req)


def test_forward_retains_anchors_and_clears_them_on_plain_outputs():
    from types import SimpleNamespace

    from freetoken.scheduler.scheduler import ForwardInput, Scheduler

    sched, _head = _draft_sched(torch.zeros(2, 64, dtype=torch.int32), torch.zeros(32))
    sched.toolcall_anchor_id = None
    sched.decode_manager = SimpleNamespace(filter_reqs=lambda reqs: None)
    sched._last_target_hidden = {"stale": torch.zeros(1)}
    batch = SimpleNamespace(spec_active=False, is_prefill=False, reqs=[], input_ids=None)
    in_map = (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
    out_map = (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
    forward_input = ForwardInput(
        batch=batch, sample_args=None, input_tuple=in_map, write_tuple=out_map)
    fresh = {9: torch.zeros(4)}
    fwd = SimpleNamespace(
        next_tokens_gpu=torch.zeros(0, dtype=torch.int32), target_hidden_anchors=fresh)
    sched.engine = SimpleNamespace(
        model=SimpleNamespace(mtp=None),
        forward_batch=lambda batch, args: fwd,
    )
    Scheduler._forward(sched, forward_input)
    # merged over the stale entry (concurrent requests keep their own anchors)
    assert sched._last_target_hidden == {"stale": torch.zeros(1), 9: fresh[9]}
    # a graph replay (anchors None) drops only its own batch's entries, never
    # goes stale: here the batch is empty so nothing is dropped
    sched.engine.forward_batch = lambda batch, args: SimpleNamespace(
        next_tokens_gpu=torch.zeros(0, dtype=torch.int32), target_hidden_anchors=None)
    Scheduler._forward(sched, forward_input)
    assert sched._last_target_hidden == {"stale": torch.zeros(1), 9: fresh[9]}


def test_forward_merges_anchors_across_concurrent_requests():
    from types import SimpleNamespace

    """A prefill for a new arrival must not drop a decoding request's anchor (live
    RuntimeError: no retained anchor hidden after an interleaved prefill). Graph
    replays (no anchors) drop only their own requests' entries."""
    from freetoken.scheduler.scheduler import ForwardInput, Scheduler

    sched = SimpleNamespace(
        _mtp_enabled=False,
        token_pool=torch.zeros((2, 32), dtype=torch.int32),
        toolcall_anchor_id=None,
        cache_manager=SimpleNamespace(spec_snapshot=lambda req: None),
        decode_manager=SimpleNamespace(filter_reqs=lambda reqs: None),
    )
    sched._last_target_hidden = {1: "anchor-1"}
    req_new = SimpleNamespace(uid=2, spec_depth=0)
    batch = SimpleNamespace(
        reqs=[req_new], spec_rows=None, spec_active=False, spec_finalized=False,
        is_prefill=True)
    sched.engine = SimpleNamespace(
        forward_batch=lambda batch, args: SimpleNamespace(
            next_tokens_gpu=torch.tensor([9], dtype=torch.int32),
            target_hidden_anchors={2: "anchor-2"}))
    out_map = (torch.tensor([1]), torch.tensor([6]))
    Scheduler._forward(
        sched, ForwardInput(batch=batch, sample_args=None,
                            input_tuple=(None, None), write_tuple=out_map))
    assert sched._last_target_hidden == {1: "anchor-1", 2: "anchor-2"}

    # a graph replay (anchors None) drops only its own batch's entries
    sched.engine = SimpleNamespace(
        forward_batch=lambda batch, args: SimpleNamespace(
            next_tokens_gpu=torch.tensor([9], dtype=torch.int32),
            target_hidden_anchors=None))
    Scheduler._forward(
        sched, ForwardInput(batch=batch, sample_args=None,
                            input_tuple=(None, None), write_tuple=out_map))
    assert sched._last_target_hidden == {1: "anchor-1"}
