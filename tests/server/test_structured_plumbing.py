"""Wire -> GenSpec plumbing for structured outputs.

The engine half (grammar compile + masked sampling) is covered by
tests/engine/test_structured_decode.py; this file pins what the OpenAI layer
promises: the neutral spec reaches SamplingParams, thinking is turned off for
structured requests, and an explicit thinking ask is named as a conflict.
"""

from __future__ import annotations

import pytest
from freetoken.server.model_meta import THINKING_OFF_KWARGS, thinking_toggle_kwargs
from freetoken.server.openai_api import (
    ChatCompletionRequest,
    _explicit_thinking_request,
    chat_request_to_genspec,
)
from freetoken.server.structured_format import normalize_response_format

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


def _request(**extra):
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "r", "strict": True, "schema": SCHEMA},
        },
    }
    body.update(extra)
    return ChatCompletionRequest(**body)


def test_structured_spec_reaches_sampling_params_and_disables_thinking():
    spec = chat_request_to_genspec(
        _request(), {}, structured_output=normalize_response_format(_request().response_format)
    )
    assert spec.sampling_params.structured_output == {
        "type": "json_schema",
        "name": "r",
        "strict": True,
        "schema": SCHEMA,
    }
    for key, value in thinking_toggle_kwargs(False).items():
        assert spec.chat_template_kwargs.get(key) == value
    assert spec.chat_template_kwargs["enable_thinking"] is False
    assert THINKING_OFF_KWARGS["enable_thinking"] is False


def test_plain_request_keeps_its_template_kwargs_untouched():
    req = _request(response_format=None, chat_template_kwargs={"custom": 1})
    spec = chat_request_to_genspec(req, {})
    assert spec.sampling_params.structured_output is None
    assert spec.chat_template_kwargs == {"custom": 1}


def test_unrelated_template_kwargs_ride_along_with_thinking_off():
    req = _request(chat_template_kwargs={"custom": 1})
    spec = chat_request_to_genspec(
        req, {}, structured_output=normalize_response_format(req.response_format)
    )
    assert spec.chat_template_kwargs["custom"] == 1
    assert spec.chat_template_kwargs["enable_thinking"] is False


@pytest.mark.parametrize(
    "extra",
    [
        {"reasoning_effort": "xhigh"},
        {"reasoning_effort": "medium"},
        {"thinking": {"type": "enabled"}},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"chat_template_kwargs": {"reasoning_effort": "high"}},
        {"chat_template_kwargs": {"thinking": True}},
        {"chat_template_kwargs": {"thinking_mode": "enabled"}},
    ],
)
def test_explicit_thinking_is_reported_as_a_conflict(extra):
    assert _explicit_thinking_request(_request(**extra)) is not None


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"reasoning_effort": "none"},
        {"reasoning_effort": "off"},
        {"thinking": {"type": "disabled"}},
        {"chat_template_kwargs": {"enable_thinking": False}},
        {"chat_template_kwargs": {"reasoning_effort": "none"}},
    ],
)
def test_thinking_off_or_unset_is_not_a_conflict(extra):
    assert _explicit_thinking_request(_request(**extra)) is None
