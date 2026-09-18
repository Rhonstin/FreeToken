"""Scheduling-policy tests for --prefill-interleave.

The managers are stubbed: _pick_next_batch touches only prefill_manager/decode_manager
runnability, the two schedule_next_batch() calls and the turn flag -- so the policy is
tested without a cache, an engine or a device.
"""

from __future__ import annotations

import types

from freetoken.scheduler.scheduler import Scheduler


class _PM:
    def __init__(self, chunks: int):
        self.chunks = chunks

    @property
    def runnable(self) -> bool:
        return self.chunks > 0

    def schedule_next_batch(self, budget: int):
        if self.chunks <= 0:
            return None
        self.chunks -= 1
        return ("prefill", self.chunks)


class _DM:
    def __init__(self, runnable: bool = True, empty_calls: int = 0):
        self._runnable = runnable
        self.empty_calls = empty_calls  # first N schedule calls yield None (e.g. no pages)

    @property
    def runnable(self) -> bool:
        return self._runnable

    def schedule_next_batch(self):
        if self.empty_calls > 0:
            self.empty_calls -= 1
            return None
        return "decode" if self._runnable else None


def _sched(interleave: bool, chunks: int, dm: _DM | None = None):
    return types.SimpleNamespace(
        config=types.SimpleNamespace(prefill_interleave=interleave),
        prefill_manager=_PM(chunks),
        decode_manager=dm or _DM(),
        prefill_budget=8192,
        _decode_turn=False,
    )


def test_off_keeps_strict_prefill_priority():
    s = _sched(False, 3)
    picks = [Scheduler._pick_next_batch(s) for _ in range(4)]
    assert picks == [("prefill", 2), ("prefill", 1), ("prefill", 0), "decode"]
    assert s._decode_turn is False


def test_on_alternates_chunk_and_decode():
    s = _sched(True, 3)
    picks = [Scheduler._pick_next_batch(s) for _ in range(6)]
    assert picks == [("prefill", 2), "decode", ("prefill", 1), "decode", ("prefill", 0), "decode"]


def test_failed_decode_attempt_falls_back_without_losing_the_turn():
    # Iteration 2's decode attempt yields None (no pages this step): the chunk runs and
    # the turn SURVIVES (True), so decode slips in on iteration 3 -- the very first
    # chance -- instead of starving for the whole prompt.
    dm = _DM(empty_calls=1)
    s = _sched(True, 2, dm)
    picks = [Scheduler._pick_next_batch(s) for _ in range(4)]
    assert picks == [("prefill", 1), ("prefill", 0), "decode", "decode"]


def test_turn_not_set_without_running_decode():
    dm = _DM(runnable=False)
    s = _sched(True, 2, dm)
    assert Scheduler._pick_next_batch(s) == ("prefill", 1)
    assert s._decode_turn is False  # nobody was waiting; a later decode starts clean
    assert Scheduler._pick_next_batch(s) == ("prefill", 0)
    assert s._decode_turn is False


def test_new_decode_backlog_gets_a_step_before_next_chunk():
    # A decode arrives mid-prefill: it must not wait for the whole remaining prompt.
    pm = _PM(2)
    dm = _DM(runnable=False)
    s = _sched(True, 0, dm)
    s.prefill_manager, s.decode_manager = pm, dm
    assert Scheduler._pick_next_batch(s) == ("prefill", 1)  # chunk 1, no decode yet
    dm._runnable = True
    assert Scheduler._pick_next_batch(s) == ("prefill", 0)  # chunk 2 -> marks the turn
    assert s._decode_turn is True
    assert Scheduler._pick_next_batch(s) == "decode"        # slip-in before anything else
