"""Teacher-forced scoring endpoint (internal POST /v1/score) and its wire plumbing.

The route reuses the normal tokenizer/scheduler ack path: a TokenizeMsg tagged
``score_only`` becomes per-chunk UserReplies carrying ``nlls``. These tests drive the
route against a fake FrontendManager (the same shape tests/server uses) and check the
response contract: N tokens yield N-1 finite NLLs, ppl == exp(mean(nll)), and the
top-1 rate is the argmax agreement over the corpus tokens.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from freetoken.message import TokenizeMsg, UserReply
from freetoken.server.control_api import register_control_routes


class FakeState:
    def __init__(self, replies: list[UserReply]) -> None:
        self.replies = replies
        self.sent: list[TokenizeMsg] = []
        self.new_users = 0

    def new_user(self) -> int:
        self.new_users += 1
        return 7

    async def send_one(self, msg: TokenizeMsg) -> None:
        self.sent.append(msg)

    async def wait_for_ack(self, uid: int):
        assert uid == 7
        for reply in self.replies:
            yield reply


def _client(state: FakeState) -> TestClient:
    app = FastAPI()
    register_control_routes(app, lambda: state)
    return TestClient(app)


def _scored_state(n_tokens: int, chunk: int = 4) -> FakeState:
    """Replies as the worker/scheduler would produce them: an admission reply carrying
    the prompt token count, then one NLL list per scoring chunk (finished on the last)."""
    nll = [0.1 * (i + 1) for i in range(n_tokens - 1)]
    chunks = [nll[i : i + chunk] for i in range(0, len(nll), chunk)] or [[]]
    replies = [
        UserReply(uid=7, incremental_output="", finished=False, prompt_tokens_delta=n_tokens)
    ]
    for i, part in enumerate(chunks):
        replies.append(
            UserReply(
                uid=7,
                incremental_output="",
                finished=(i == len(chunks) - 1),
                nlls=part,
                top1_hits=min(2, len(part)),
            )
        )
    return FakeState(replies)


def test_score_route_returns_nll_ppl_and_top1():
    state = _scored_state(n_tokens=9, chunk=4)
    r = _client(state).post("/v1/score", json={"text": "any text", "chunk": 4})
    assert r.status_code == 200
    body = r.json()
    assert len(body["nll"]) == body["n_tokens"] - 1 == 8
    assert all(math.isfinite(x) for x in body["nll"])
    assert body["ppl"] == pytest.approx(math.exp(sum(body["nll"]) / len(body["nll"])))
    assert body["top1_hits"] == 2 + 2
    assert body["top1_rate"] == pytest.approx(body["top1_hits"] / len(body["nll"]))
    # The request reached the tokenizer path as a scoring request with its chunk cap.
    assert len(state.sent) == 1
    msg = state.sent[0]
    assert msg.score_only is True and msg.score_chunk == 4
    assert msg.sampling_params.max_tokens == 0
    assert msg.text == "any text"


def test_score_route_default_chunk_is_1024():
    state = _scored_state(n_tokens=3)
    r = _client(state).post("/v1/score", json={"text": "abc"})
    assert r.status_code == 200
    assert state.sent[0].score_chunk == 1024


def test_score_route_empty_text_is_400():
    state = _scored_state(n_tokens=3)
    r = _client(state).post("/v1/score", json={"text": ""})
    assert r.status_code == 400
    assert state.sent == []  # rejected before any engine work


def test_score_route_chunk_out_of_range_is_422():
    state = _scored_state(n_tokens=3)
    r = _client(state).post("/v1/score", json={"text": "abc", "chunk": 0})
    assert r.status_code == 422
    r = _client(state).post("/v1/score", json={"text": "abc", "chunk": 10**9})
    assert r.status_code == 422


def test_score_route_context_length_exceeded_is_413():
    state = FakeState(
        [
            UserReply(
                uid=7,
                incremental_output="",
                finished=True,
                error="prompt is too long: 10 tokens > 5 maximum",
                error_code="context_length_exceeded",
            )
        ]
    )
    r = _client(state).post("/v1/score", json={"text": "too long"})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "context_length_exceeded"


def test_score_route_zero_token_prompt_is_400():
    state = FakeState(
        [
            UserReply(
                uid=7,
                incremental_output="",
                finished=True,
                error="prompt must contain at least one token",
            )
        ]
    )
    r = _client(state).post("/v1/score", json={"text": " "})
    assert r.status_code == 400
    assert "at least one token" in r.json()["error"]["message"]


def test_score_reply_translation_carries_nlls():
    from freetoken.message import ScoreChunkMsg
    from freetoken.tokenizer.server import _score_reply

    reply = _score_reply(
        ScoreChunkMsg(uid=3, nlls=[0.5, 1.5], top1_hits=1, finished=True)
    )
    assert reply.uid == 3
    assert reply.nlls == [0.5, 1.5]
    assert reply.top1_hits == 1
    assert reply.finished is True


def test_score_state_carries_transient_chunk_len():
    # The engine stashes the pre-completion row count on the request; the drain slices the
    # all-rows logits with it. Guard the field survives every Req construction path.
    from freetoken.core import Req
    from freetoken.core import SamplingParams
    from freetoken.scheduler.prefill import ChunkedReq

    for cls in (Req, ChunkedReq):
        req = cls(
            input_ids=torch.arange(4, dtype=torch.int32),
            table_idx=0,
            cached_len=0,
            output_len=0,
            uid=1,
            sampling_params=SamplingParams(max_tokens=0),
            cache_handle=SimpleNamespace(cached_len=0),
        )
        assert req.score_only is False and req.score_chunk_len == 0
        req.score_only = True
        req.score_chunk_len = 4
        assert req.score_chunk_len == 4
