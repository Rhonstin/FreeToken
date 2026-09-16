# Structured outputs (`response_format`)

The server enforces OpenAI **structured outputs** at the sampler: when a request
carries `response_format`, the tokens the model may emit are constrained to the
requested JSON by a grammar, so the reply is always valid JSON of the requested
shape. `openai.beta.chat.completions.parse(...)` (the OpenAI Python SDK) and any
client that sends a JSON Schema work unchanged.

Two formats are honored:

| `response_format` | Promise |
| --- | --- |
| `{"type": "json_object"}` | any JSON object |
| `{"type": "json_schema", "json_schema": {"name", "strict", "schema"}}` | a document that validates against `schema` |

Anything else (`{"type": "grammar"}`, a malformed `json_schema`, ...) is a `400`.

## Requirements

The grammar backend is the optional `xgrammar` dependency:

```bash
uv pip install "freetoken[structured]"     # or: pip install xgrammar
```

Without it, structured requests are refused with a message naming the missing
dependency; plain text requests are unaffected. A server that never receives
`response_format` never imports xgrammar.

## Example

```bash
curl http://127.0.0.1:1919/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next-NVFP4",
    "messages": [{"role": "user", "content": "Return greeting + count 3"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "r",
        "strict": true,
        "schema": {
          "type": "object",
          "additionalProperties": false,
          "required": ["answer", "count"],
          "properties": {
            "answer": {"type": "string"},
            "count": {"type": "integer"}
          }
        }
      }
    }
  }'
```

```python
from openai import OpenAI
from pydantic import BaseModel

class R(BaseModel):
    answer: str
    count: int

client = OpenAI(base_url="http://127.0.0.1:1919/v1", api_key="sk-local")
result = client.beta.chat.completions.parse(
    model="Qwen3.8-Flash-Next-NVFP4",
    messages=[{"role": "user", "content": "Return JSON answer=greeting,count=3"}],
    response_format=R,
)
print(result.choices[0].message.parsed)   # R(answer='greeting', count=3)
```

## Behavior

- The request finishes the moment the document is complete
  (`finish_reason: "stop"`); no trailing stop token is generated or streamed.
- **Thinking is off** for structured requests. The grammar constrains from the
  first sampled token, so a reasoning block could not be reproduced; an explicit
  thinking ask (`reasoning_effort` other than `none`/`off`, `thinking.type:
  "enabled"`, `chat_template_kwargs.enable_thinking`) is refused with a `400`
  rather than silently ignored.
- `tools` and `stop` are refused together with `response_format`: the grammar
  leaves no room for a tool call, and a stop string could only truncate the JSON
  the client was promised.
- `max_tokens` still bounds the reply. If the budget runs out before the
  document closes, the reply ends with `finish_reason: "length"` and is
  truncated -- give structured requests enough room (a few hundred tokens is
  plenty for a small schema).
- Streaming works: partial JSON arrives as it is generated, exactly like the
  hosted API.
- The compiled grammar is cached per schema inside each engine process, so a
  second request with the same Pydantic model skips compilation. Only
  constrained steps pay for masking; plain traffic keeps the zero-copy sampling
  path.

## Notes for operators

- One grammar matcher per request; the cache is bounded (128 schemas per engine
  process).
- Data-parallel mode compiles per engine group, independently.
- Speculative decode (MTP) is disabled for a structured request: draft rows would
  advance the matcher on tokens that may be rejected.
