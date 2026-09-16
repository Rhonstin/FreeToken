"""Parsing and validation of OpenAI ``response_format`` (structured outputs).

Kept free of torch/xgrammar imports so the HTTP frontend can validate a request
cheaply, on the event loop, without pulling the engine's dependencies in or
compiling anything. The compiled grammar lives in ``freetoken.engine.structured``.
"""

from __future__ import annotations

from typing import Any

#: response_format types this server can honor. Everything else is a client error.
SUPPORTED_TYPES = ("text", "json_object", "json_schema")


def normalize_response_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the neutral spec for a supported ``response_format``, or None for plain text.

    The neutral spec is what rides on ``SamplingParams.structured_output`` across the
    frontend -> tokenizer -> scheduler IPC and what the engine compiles into a grammar:

        {"type": "json_object"}
        {"type": "json_schema", "name": str, "strict": bool, "schema": {...}}

    Raises ``ValueError`` for a malformed or unsupported format so the API can answer a
    clean 400 instead of failing later inside the engine.
    """
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    fmt = response_format.get("type")
    if fmt in (None, "text"):
        return None
    if fmt == "json_object":
        return {"type": "json_object"}
    if fmt != "json_schema":
        raise ValueError(
            f"response_format type {fmt!r} is not supported; "
            f"use {'/'.join(repr(t) for t in SUPPORTED_TYPES)}"
        )
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        raise ValueError("response_format.json_schema must be an object")
    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("response_format.json_schema.schema must be a JSON Schema object")
    name = json_schema.get("name")
    return {
        "type": "json_schema",
        "name": name if isinstance(name, str) and name else "response",
        # OpenAI's `strict` is a promise about the schema's shape, not a switch that is
        # safe to assume: pass it through so a non-strict schema still compiles.
        "strict": bool(json_schema.get("strict", False)),
        "schema": schema,
    }


def structured_conflict(*, tools: Any, stop: Any, ignore_eos: bool = False) -> str | None:
    """Why a structured request cannot be served as asked, or None if it can.

    The grammar constrains every sampled token, so a tool call or a client stop string
    could only ever truncate the JSON the client is promised. OpenAI's own structured
    outputs reject the same combinations rather than degrade them. ``ignore_eos`` is
    rejected for a different reason: a completed grammar is closed by the model's stop
    token, and a client that ignores EOS would keep the request decoding past a complete
    document instead of receiving it.
    """
    if tools:
        return "tools are not supported together with response_format"
    if stop:
        return "stop is not supported together with response_format"
    if ignore_eos:
        return "ignore_eos is not supported together with response_format"
    return None
