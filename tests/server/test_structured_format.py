"""OpenAI response_format parsing (the wire half of structured outputs).

The engine half -- compiling the schema into an xgrammar grammar and masking
logits against it -- lives in tests/engine/test_structured_decode.py.
"""

from __future__ import annotations

import pytest
from freetoken.server.structured_format import (
    normalize_response_format,
    structured_conflict,
)


def test_unset_and_text_mean_plain_output():
    assert normalize_response_format(None) is None
    assert normalize_response_format({"type": "text"}) is None
    assert normalize_response_format({}) is None


def test_json_object_normalizes_to_a_bare_spec():
    assert normalize_response_format({"type": "json_object"}) == {"type": "json_object"}


def test_json_schema_keeps_name_strict_and_schema():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "count"],
        "properties": {"answer": {"type": "string"}, "count": {"type": "integer"}},
    }
    spec = normalize_response_format(
        {"type": "json_schema", "json_schema": {"name": "r", "strict": True, "schema": schema}}
    )
    assert spec == {"type": "json_schema", "name": "r", "strict": True, "schema": schema}
    # A missing name is named by the server; a missing strict is not strict. Both stay
    # JSON-only, the property the engine relies on.
    assert normalize_response_format(
        {"type": "json_schema", "json_schema": {"schema": schema}}
    ) == {"type": "json_schema", "name": "response", "strict": False, "schema": schema}


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "grammar"},
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {}},
        {"type": "json_schema", "json_schema": {"schema": []}},
        "json_object",
        [],
    ],
)
def test_malformed_formats_raise_value_error(response_format):
    with pytest.raises(ValueError):
        normalize_response_format(response_format)


def test_structured_conflicts_name_the_offending_field():
    assert structured_conflict(tools=None, stop=None) is None
    assert structured_conflict(tools=[{"type": "function"}], stop=None)
    assert structured_conflict(tools=None, stop=["\n\n"])
    # An empty stop is dropped by resolve_sampling before it ever reaches the engine.
    assert structured_conflict(tools=None, stop="") is None
