"""Teacher-forced scoring in the scheduler: NLL math, chunk-boundary handling, the
prefill chunk cap, generation/scoring isolation, and prefix-cache discard.

These are CPU tests: the engine's all-rows logits are faked, so the whole scheduler-side
scoring path (chunk scheduling -> NLL drain -> reply -> free) runs without a GPU. The
``ParallelLMHead`` all-rows projection itself is GPU-only and is validated on hardware.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

CHUNK_BUDGET = 8
WIDTH = 64
MAX_RUNNING = 4


def _setup_context() -> None:
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))


def _build_managers(num_pages: int = 64):
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    _setup_context()
    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
    cm = CacheManager(num_pages=num_pages, page_size=1, page_table=pt, type="radix")
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    return cm, tm, pm


def _score_pending(uid: int, prompt_len: int, score_chunk: int):
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    return PendingReq(
        uid=uid,
        input_ids=torch.arange(prompt_len, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=0),
        score_only=True,
        score_chunk=score_chunk,
    )


def _drain_double(cm, tm):
    """A Scheduler-shaped double exposing only what the scoring drain touches."""
    from freetoken.scheduler.scheduler import Scheduler

    class Drain:
        def __init__(self) -> None:
            self.cache_manager = cm
            self.table_manager = tm
            self.replies: list = []

        def send_result(self, replies) -> None:
            self.replies.extend(replies)

    Drain._free_req_resources = Scheduler._free_req_resources
    Drain._process_score_batch = Scheduler._process_score_batch
    Drain._score_req_chunk = Scheduler._score_req_chunk
    return Drain()


def _forward_one(pm, cm, batch, logits_factory=None):
    """Allocate + complete one scheduled chunk, then drain it; returns the batch."""
    cm.allocate_paged(batch.reqs)
    n_rows = 0
    for req in batch.reqs:
        n_rows += req.extend_len
    for req in batch.reqs:
        req.score_chunk_len = req.extend_len
        req.complete_n(req.extend_len)
    factory = logits_factory or (lambda rows, reqs: torch.zeros(rows, 16))
    fake = SimpleNamespace(spec_logits=factory(n_rows, batch.reqs))
    return batch, fake


# --------------------------------------------------------------------------- #
# NLL math
# --------------------------------------------------------------------------- #
def test_nll_matches_log_softmax_and_top1():
    from freetoken.scheduler.scheduler import _score_nlls_from_logits

    torch.manual_seed(0)
    logits = torch.randn(5, 7)
    targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    nlls, hits = _score_nlls_from_logits(logits, targets)
    ref = -torch.log_softmax(logits.float(), dim=-1).gather(
        1, targets.long().unsqueeze(1)
    ).squeeze(1)
    assert nlls == pytest.approx(ref.tolist())
    assert hits == int((logits.argmax(dim=-1) == targets).sum())

    # Blocking is a memory knob only: identical results at any block size.
    for block in (1, 2, 5):
        blocked, blocked_hits = _score_nlls_from_logits(logits, targets, block=block)
        assert blocked == pytest.approx(nlls)
        assert blocked_hits == hits


def test_clean_text_has_lower_ppl_than_scrambled():
    """The definitional check: a deterministic tiny transition model scores a
    high-probability continuation loop below a low-probability scrambled walk."""
    from freetoken.scheduler.scheduler import _score_nlls_from_logits

    vocab = 4
    stay, jump = 0.7, 0.1
    transition = torch.full((vocab, vocab), jump)
    transition.diagonal().fill_(stay)

    def ppl(sequence: list[int]) -> float:
        seq = torch.tensor(sequence)
        logits = torch.log(transition[seq[:-1]])
        nlls, _ = _score_nlls_from_logits(logits, seq[1:])
        assert len(nlls) == len(sequence) - 1
        return math.exp(sum(nlls) / len(nlls))

    clean = [0] * 33
    scrambled = [i % vocab for i in range(33)]
    assert ppl(clean) == pytest.approx(1.0 / stay, rel=1e-4)
    assert ppl(scrambled) == pytest.approx(1.0 / jump, rel=1e-4)
    assert ppl(clean) < ppl(scrambled)


# --------------------------------------------------------------------------- #
# Prefill scheduling: cap + isolation
# --------------------------------------------------------------------------- #
def test_score_chunk_cap_splits_prompt_and_isolates_from_generation():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    _cm, _tm, pm = _build_managers()
    pm.pending_list = [
        PendingReq(1, torch.arange(8, dtype=torch.int32), SamplingParams(max_tokens=4)),
        _score_pending(2, 10, score_chunk=4),
    ]

    normal = pm.schedule_next_batch(CHUNK_BUDGET)
    assert normal is not None and normal.score_only is False
    assert [r.uid for r in normal.reqs] == [1]  # never mixed with the scoring request

    sizes = []
    while pm.runnable:
        batch = pm.schedule_next_batch(CHUNK_BUDGET)
        assert batch is not None and batch.score_only is True
        assert [r.uid for r in batch.reqs] == [2]
        sizes.append(batch.reqs[0].extend_len)
        for req in batch.reqs:
            req.complete_one()
    assert sizes == [4, 4, 2]  # capped at score_chunk, not the 8-row budget


def test_score_request_skips_prefix_match():
    cm, _tm, pm = _build_managers()
    # Warm the prefix cache with a normal request over the same tokens.
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    pm.pending_list = [
        PendingReq(1, torch.arange(8, dtype=torch.int32), SamplingParams(max_tokens=1))
    ]
    batch = pm.schedule_next_batch(CHUNK_BUDGET)
    assert batch is not None
    cm.allocate_paged(batch.reqs)
    for req in batch.reqs:
        req.complete_one()
    cm.cache_req(batch.reqs[0], finished=True)

    pm.pending_list = [_score_pending(2, 8, score_chunk=8)]
    batch = pm.schedule_next_batch(CHUNK_BUDGET)
    assert batch is not None and batch.score_only is True
    assert batch.prompt_admissions == [(2, 8, 0)]  # no cached tokens credited
    assert batch.reqs[0].cache_handle.cached_len == 0


# --------------------------------------------------------------------------- #
# Drain: chunk replies, boundary scoring, discard-on-free
# --------------------------------------------------------------------------- #
def test_score_drain_scores_boundaries_and_discards_prefix():
    cm, tm, pm = _build_managers()
    drain = _drain_double(cm, tm)
    prompt_len = 10
    pm.pending_list = [_score_pending(7, prompt_len, score_chunk=4)]

    batches = []
    while pm.runnable:
        batch = pm.schedule_next_batch(CHUNK_BUDGET)
        assert batch is not None and batch.score_only is True
        # Flat zeros over a 16-token vocab: logsumexp = log(16), so every row's NLL is
        # exactly log(16), and only target 0 (argmax of a zero row) counts as a hit.
        batch, fake = _forward_one(pm, cm, batch)
        batches.append((batch, fake))
        drain._process_score_batch(batch, fake)

    # Chunks [4, 4, 2]: the final chunk's last row has no successor, so it scores 1 row.
    assert [len(m.nlls) for m in drain.replies] == [4, 4, 1]
    assert [m.finished for m in drain.replies] == [False, False, True]
    flat = [x for m in drain.replies for x in m.nlls]
    assert flat == pytest.approx([math.log(16)] * (prompt_len - 1))
    # Targets are the successor ids (input_ids sequential), so only position 0's target
    # (token 1) differs from the argmax-0; hits are counted per chunk.
    assert sum(m.top1_hits for m in drain.replies) == 0

    # The request was released WITHOUT seeding the prefix cache.
    req = batches[-1][0].reqs[0]
    assert req.table_idx == -1
    assert cm.prefix_cache.size_info.evictable_size == 0
    assert len(cm.free_slots) == cm.num_pages
    cm.check_integrity()


def test_score_drain_boundary_target_comes_from_next_chunk():
    """A chunk's last row predicts the FIRST token of the next chunk -- the boundary
    target is read from score_full_ids, not from the chunk's own input_ids."""
    cm, tm, pm = _build_managers()
    drain = _drain_double(cm, tm)
    prompt_len = 6
    pm.pending_list = [_score_pending(9, prompt_len, score_chunk=4)]

    seen: list[tuple[int, int]] = []

    def factory(rows, reqs):
        # Vocabulary 8; make row r's argmax equal the global successor position so the
        # top-1 count proves the target slicing: target at position p is token p+1 == p+1.
        logits = torch.full((rows, 8), -10.0)
        offset = 0
        for req in reqs:
            start = req.cached_len - req.score_chunk_len
            for r in range(req.score_chunk_len):
                pos = start + r
                if pos + 1 < prompt_len:
                    logits[offset, pos + 1] = 0.0
                offset += 1
        return logits

    while pm.runnable:
        batch = pm.schedule_next_batch(CHUNK_BUDGET)
        assert batch is not None
        batch, fake = _forward_one(pm, cm, batch, logits_factory=factory)
        seen.append((batch.reqs[0].cached_len - batch.reqs[0].score_chunk_len,
                     batch.reqs[0].cached_len))
        drain._process_score_batch(batch, fake)

    assert seen == [(0, 4), (4, 6)]
    assert [len(m.nlls) for m in drain.replies] == [4, 1]
    # Row 3 (chunk 0) predicts token 4, which lives only in the NEXT chunk's input:
    # an exact top-1 hit there is only possible if the boundary target slice is right.
    assert sum(m.top1_hits for m in drain.replies) == prompt_len - 1


def test_scoring_abort_frees_without_reply():
    from freetoken.message import ScoreChunkMsg

    cm, tm, pm = _build_managers()
    drain = _drain_double(cm, tm)
    pm.pending_list = [_score_pending(11, 6, score_chunk=4)]
    batch = pm.schedule_next_batch(CHUNK_BUDGET)
    assert batch is not None
    batch, fake = _forward_one(pm, cm, batch)
    batch.reqs[0].aborted = True
    drain._process_score_batch(batch, fake)
    assert drain.replies == []
    assert batch.reqs[0].table_idx == -1
    assert len(cm.free_slots) == cm.num_pages
    cm.check_integrity()
    assert ScoreChunkMsg is not None  # imported for the reply-type contract above
