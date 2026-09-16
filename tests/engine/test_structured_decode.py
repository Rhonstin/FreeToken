"""Grammar-constrained decoding: schema -> matcher -> masked logits (CPU).

The tokenizer here is a hand-built single-character vocabulary so the walk is
exact and readable: every token is one JSON character, which keeps the expected
sequence obvious and lets the test state its own expected values.
"""

from __future__ import annotations

import torch

import pytest

xgr = pytest.importorskip("xgrammar", reason="structured output needs the xgrammar extra")

from freetoken.engine.structured import (  # noqa: E402
    StructuredCompiler,
    advance_states,
    cache_key,
    mask_logits,
    states_of,
)

# {" a 1 } plus a stop token. RAW vocab: token text is the literal bytes.
VOCAB = ["{", "}", '"', ":", ",", " ", "a", "b", "1", "2", "3", "[", "]", "<eos>"]
EOS = VOCAB.index("<eos>")
TOKEN = {text: i for i, text in enumerate(VOCAB)}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["a"],
    "properties": {"a": {"type": "integer"}},
}


def _compiler() -> StructuredCompiler:
    info = xgr.TokenizerInfo(VOCAB, xgr.VocabType.RAW, stop_token_ids=[EOS])
    return StructuredCompiler(xgr.GrammarCompiler(info), info, stop_token_ids=[EOS])


def _walk(state, target, bias=10.0):
    """Greedy walk biased toward ``target`` strings, one character per step.

    The mask is the only thing standing between the bias and the produced text, so
    every step asserts the promise directly: a token the grammar forbids can never
    win the argmax no matter how hard it is biased.
    """
    produced: list[int] = []
    for step in range(len(target) + 1):
        logits = torch.full((1, len(VOCAB)), -1.0)
        if step < len(target):
            logits[0, TOKEN[target[step]]] = bias
        masked = mask_logits(logits, [state])
        token = int(masked.argmax(dim=-1).item())
        assert state.accept(token), f"sampled token {token} rejected by the grammar"
        produced.append(token)
        if state.terminated:
            break
    return "".join(VOCAB[t] for t in produced)


def test_walk_emits_exactly_the_reference_document():
    spec = {"type": "json_schema", "name": "r", "strict": True, "schema": SCHEMA}
    state = _compiler().new_state(spec)
    assert _walk(state, '{"a": 1}') == '{"a": 1}'
    assert state.terminated


def test_mask_never_allows_a_schema_violation():
    # After `{"a": ` the schema requires an integer, so a biased quote must lose to a digit.
    state = _compiler().new_state({"type": "json_schema", "schema": SCHEMA})
    assert state.accept(TOKEN["{"]), "object open must be allowed"
    assert state.accept(TOKEN['"']) and state.accept(TOKEN["a"]) and state.accept(TOKEN['"'])
    assert state.accept(TOKEN[":"])
    logits = torch.full((1, len(VOCAB)), -1.0)
    logits[0, TOKEN['"']] = 100.0
    masked = mask_logits(logits, [state])
    assert int(masked.argmax(dim=-1)) != TOKEN['"'], "string token leaked into an integer slot"


def test_mask_is_skipped_and_the_tensor_untouched_when_nothing_is_constrained():
    state = _compiler().new_state({"type": "json_schema", "schema": SCHEMA})
    logits = torch.zeros(2, len(VOCAB))
    # No states -> identity, no clone.
    assert mask_logits(logits, [None, None]) is logits
    # With a state -> the input stays untouched (the engine hands out a clone).
    masked = mask_logits(logits, [state, None])
    assert masked is not logits
    assert torch.isfinite(logits).all()
    assert not torch.isfinite(masked[0][TOKEN["b"]]), "row 0 must be masked"
    assert torch.isfinite(masked[1]).all(), "row 1 has no grammar"


def test_states_of_reads_req_attributes():
    class Req:
        def __init__(self, state):
            self.structured_state = state

    state = _compiler().new_state({"type": "json_schema", "schema": SCHEMA})
    assert states_of([Req(None), Req(state)]) == [None, state]


def test_compiled_grammars_are_shared_per_schema_and_split_per_schema():
    compiler = _compiler()
    compiler.new_state({"type": "json_schema", "schema": SCHEMA})
    compiler.new_state({"type": "json_schema", "schema": SCHEMA})
    assert len(compiler._grammars) == 1
    compiler.new_state({"type": "json_object"})
    compiler.new_state({"type": "json_object"})
    assert len(compiler._grammars) == 2
    assert cache_key({"type": "json_object"}) == "json_object"


def test_json_object_accepts_any_object_and_rejects_junk():
    state = _compiler().new_state({"type": "json_object"})
    assert _walk(state, '{"b": 1}', bias=1.0).startswith('{"b"')
    assert state.terminated
    other = _compiler().new_state({"type": "json_object"})
    logits = torch.full((1, len(VOCAB)), -1.0)
    logits[0, TOKEN[","]] = 100.0  # a leading comma can never start a JSON value
    masked = mask_logits(logits, [other])
    assert int(masked.argmax(dim=-1)) != TOKEN[","]


def test_terminated_state_masks_down_to_stop_tokens_only():
    # Overlap scheduling can launch one more step after a grammar completed; that step must
    # only be able to emit a stop token (the detokenizer drops it), never a real one.
    state = _compiler().new_state({"type": "json_object"})
    assert _walk(state, "{}") == "{}"
    assert state.terminated
    logits = torch.full((1, len(VOCAB)), -1.0)
    logits[0, TOKEN["a"]] = 100.0
    masked = mask_logits(logits, [state])
    allowed = [text for i, text in enumerate(VOCAB) if torch.isfinite(masked[0][i])]
    assert allowed == ["<eos>"], allowed


def test_terminated_state_masks_down_to_stop_tokens_only():
    # Overlap scheduling can launch one more step after a grammar completed; that step must
    # only be able to emit a stop token (the detokenizer drops it), never a real one.
    state = _compiler().new_state({"type": "json_object"})
    assert _walk(state, "{}") == "{}"
    assert state.terminated
    logits = torch.full((1, len(VOCAB)), -1.0)
    logits[0, TOKEN["a"]] = 100.0
    masked = mask_logits(logits, [state])
    allowed = [text for i, text in enumerate(VOCAB) if torch.isfinite(masked[0][i])]
    assert allowed == ["<eos>"], allowed


def test_state_records_the_position_of_the_completing_token():
    # The scheduler finishes a structured request on the token at this position, because the
    # engine has already advanced the matcher past the batch being drained (overlap).
    state = _compiler().new_state({"type": "json_schema", "schema": SCHEMA})
    for ch in '{"a": 1}':
        assert state.accept(TOKEN[ch])
    assert state.terminated
    assert state.advanced == 8  # every character of the document, one per token
    assert state.completed_at == 8  # the closing brace is the 8th (1-based) token
    # A further accepted token must not move it: extra steps cannot re-finish the request.
    assert state.accept(TOKEN["b"])
    assert state.completed_at == 8


def test_advance_states_feeds_the_sampled_token_into_the_matcher_and_skips_plain_rows():
    class Req:
        def __init__(self, state, uid=0):
            self.structured_state = state
            self.uid = uid

    state = _compiler().new_state({"type": "json_schema", "schema": SCHEMA})
    # A CPU tensor stands in for the sampled-token GPU tensor: advance_states only reads it.
    advance_states(
        [Req(None, uid=1), Req(state)],
        torch.tensor([TOKEN["b"], TOKEN["{"]], dtype=torch.int32),
    )
    # The constrained row consumed its '{': a second one is no longer a valid continuation.
    logits = torch.full((1, len(VOCAB)), -1.0)
    logits[0, TOKEN["{"]] = 100.0
    masked = mask_logits(logits, [state])
    assert int(masked.argmax(dim=-1)) != TOKEN["{"]

    # A terminated state ignores further tokens instead of double-advancing.
    assert _walk(state, '"a": 1}') == '"a": 1}'
    assert state.terminated
    advance_states([Req(state)], torch.tensor([TOKEN["b"]], dtype=torch.int32))
