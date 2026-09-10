"""Multi-token (speculative) drains deliver several DetokenizeMsgs for one uid in a
single batch; the detokenizer must emit one clean delta per message.

Live regression: with the batch_decode two-loop bookkeeping, the second message's
slice still covered the previous token, so the stream showed " need" -> " need answer"
(the client then concatenated the repeat).
"""

from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager

# token id -> text (space-prefixed, like BPE word tokens)
_VOCAB = {5: "We", 6: " need", 7: " answer", 8: " step", 9: ""}


class _StubTokenizer:
    eos_token_id = 9

    def decode(self, ids, **_kwargs) -> str:
        return "".join(_VOCAB[i] for i in ids)

    def batch_decode(self, rows, **_kwargs):
        return [self.decode(r) for r in rows]


def _msg(uid: int, tok: int, *, finished: bool = False) -> DetokenizeMsg:
    return DetokenizeMsg(
        uid=uid, next_token=tok, finished=finished, finish_reason=None,
        matched_stop=None, stop_strs=None,
    )


def test_two_messages_for_one_uid_emit_two_clean_deltas():
    mgr = DetokenizeManager(_StubTokenizer())
    out = mgr.detokenize([_msg(0, 5), _msg(0, 6)])
    assert out[:2] == ["We", " need"]
    out = mgr.detokenize([_msg(0, 7), _msg(0, 8)])
    assert out == [" answer", " step"]


def test_interleaved_uids_keep_independent_state():
    mgr = DetokenizeManager(_StubTokenizer())
    out = mgr.detokenize([_msg(0, 5), _msg(1, 5), _msg(0, 6), _msg(1, 7)])
    assert out == ["We", "We", " need", " answer"]


def test_single_message_batches_are_unchanged():
    mgr = DetokenizeManager(_StubTokenizer())
    assert mgr.detokenize([_msg(0, 5)]) == ["We"]
    assert mgr.detokenize([_msg(0, 6)]) == [" need"]
    assert mgr.detokenize([_msg(0, 7)]) == [" answer"]


def test_finished_eos_message_flushes_and_drops_state():
    mgr = DetokenizeManager(_StubTokenizer(), eos_token_ids=frozenset({9}))
    assert mgr.detokenize([_msg(0, 5)]) == ["We"]
    assert mgr.detokenize([_msg(0, 9, finished=True)]) == [""]
    assert 0 not in mgr.decode_map
    assert mgr.detokenize([_msg(0, 5)]) == ["We"]  # fresh state after finish
