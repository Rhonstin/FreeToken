from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from freetoken.core import SCORE_CHUNK_MAX


class MessageContent(BaseModel):
    type: str
    text: str | None = None
    image_url: Any | None = None
    audio_url: Any | None = None


class Function(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function


class ToolChoiceFunction(BaseModel):
    name: str


class ToolChoiceObject(BaseModel):
    type: Literal["function"] = "function"
    function: ToolChoiceFunction


class StreamOptions(BaseModel):
    include_usage: bool = False


class FunctionCall(BaseModel):
    name: str | None = None
    arguments: str | dict[str, Any] | None = None


class ToolCall(BaseModel):
    id: str | None = None
    index: int | None = None
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    role: str
    content: str | list[MessageContent] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None
    reasoning_content: str | None = None
    thinking: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    reasoning_effort: str | None = None
    # DeepSeek-wire thinking toggle ({"type": "enabled"|"disabled"}). Any so a
    # foreign shape stays ignored (extra="allow" swallowed it before this field
    # existed) instead of becoming a bare 422 at the route boundary; the handler
    # reads the dict form and 400s only on an unknown "type" value.
    thinking: Any | None = None
    ignore_eos: bool = False
    tools: list[Tool] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoiceObject | None = None
    parallel_tool_calls: bool | None = None
    function_call: Any | None = None
    logit_bias: dict[str, float] | None = None
    response_format: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _sync_max_completion_tokens(self) -> "ChatCompletionRequest":
        if self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    logprobs: int | None = None
    echo: bool = False
    suffix: str | None = None
    logit_bias: dict[str, float] | None = None
    response_format: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _sync_max_completion_tokens(self) -> "CompletionRequest":
        if self.max_completion_tokens is not None:
            self.max_tokens = self.max_completion_tokens
        return self


# Teacher-forced scoring (internal ``POST /v1/score``). ``chunk`` caps the prefill rows
# per forward; the all-rows logits retire at the end of each chunk. At 1024 rows and a
# 248k vocab the retained logits are ~0.5 GiB bf16 / ~1 GiB fp32, so the cap is a memory
# guard, not a throughput knob. SCORE_CHUNK_MAX (core) is the hard ceiling the scheduler
# also enforces.
SCORE_CHUNK_DEFAULT = 1024


class ScoreRequest(BaseModel):
    text: str
    chunk: int = Field(default=SCORE_CHUNK_DEFAULT, ge=1, le=SCORE_CHUNK_MAX)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "FreeToken"
    root: str
    # The model's own limit, not the KV budget in force. Two spellings of the same number:
    # `max_model_len` is vLLM/SGLang's, `context_length` what most other clients look for.
    max_model_len: int | None = None
    context_length: int | None = None
    # The checkpoint's probed effort vocabulary (freetoken.tokenizer.effort); None
    # (not []) when the model has no effort knob or the probe could not run.
    supported_reasoning_efforts: list[str] | None = None
    default_reasoning_effort: str | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)
