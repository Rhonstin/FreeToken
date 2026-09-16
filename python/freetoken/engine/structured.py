"""xgrammar-backed grammar-constrained decoding (OpenAI structured outputs).

Where this sits in one engine step:

    Scheduler._process_one_msg   -> StructuredCompiler.new_state(spec)   (admission, per request)
    Scheduler._prepare_batch     -> nothing (the state rides on Req)
    Engine.forward_batch         -> mask_logits(logits, states)          (before sampling)
    Scheduler._process_last_data -> state.accept(token)                  (after sampling)

Correctness notes:

* ``mask_logits`` CLONES the batch logits when any row is constrained. The tensor it is
  handed is a view of the CUDA-graph logits buffer / lm_head output; the same tensor also
  feeds the speculative and scoring paths, and a graph replay must not find -inf holes
  from a previous step. With no constrained row the input passes through unchanged, so
  plain traffic keeps its zero-copy path.
* A matcher that has accepted its final token reports ``is_terminated()``; xgrammar then
  refuses another ``fill_next_token_bitmask`` call. Terminated states are never asked for
  a mask again -- the scheduler finishes the request with ``finish_reason="stop"`` after
  streaming the token that completed the document.
* One compiled grammar is shared by every request with the same schema (the cache key is
  the schema JSON), but each request gets its own matcher: matchers are stateful.
* xgrammar is optional. Without it ``available()`` is False and the API refuses structured
  requests with a message naming the missing dependency instead of failing in the engine.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Sequence

import torch

try:  # optional dependency: the engine serves plain traffic without it
    import xgrammar as xgr
except Exception:  # pragma: no cover - exercised on installs without xgrammar
    xgr = None

logger = logging.getLogger(__name__)

#: How many compiled grammars one process keeps. Pydantic models in one application are
#: few; the bound only stops a hostile/buggy client from pinning arbitrary schemas.
MAX_CACHED_GRAMMARS = 128


def available() -> bool:
    """Whether the grammar backend is importable in this process."""
    return xgr is not None


def cache_key(spec: dict[str, Any]) -> str:
    """Stable key for one neutral spec (see server/structured_format.normalize_response_format)."""
    if spec.get("type") == "json_object":
        return "json_object"
    schema = spec.get("schema") or {}
    return "json_schema:" + json.dumps(schema, sort_keys=True, separators=(",", ":"))


@dataclass
class StructuredState:
    """One request's live matcher. Created at admission, advanced right after sampling.

    ``stop_token_ids`` are the ids a terminated state is allowed to emit: overlap
    scheduling can have already launched one more step for a request whose grammar just
    completed, and that extra step must produce a stop token (dropped by the detokenizer)
    rather than a harmless-looking token that would land in the client's JSON.
    """

    matcher: Any
    spec: dict[str, Any]
    stop_token_ids: tuple[int, ...] = ()
    terminated: bool = False
    #: Tokens this matcher has consumed (engine-side count) and the 1-based index of the
    #: token that completed the document. The drain compares the latter against the
    #: position of the token it is streaming to decide the finish: the engine runs one
    #: batch ahead of the drain, so the drain cannot read `terminated` off the matcher.
    advanced: int = 0
    completed_at: int | None = None

    def accept(self, token_id: int) -> bool:
        """Feed one sampled token to the matcher; True when the grammar is now complete.

        Returns False when xgrammar rejects the token. That cannot happen while the mask
        from this same step was applied, so the caller treats it as an engine invariant
        violation rather than a client error.
        """
        if self.terminated:
            return True
        if not self.matcher.accept_token(int(token_id)):
            return False
        self.advanced += 1
        if self.matcher.is_terminated():
            self.terminated = True
            self.completed_at = self.advanced
        return True


class StructuredCompiler:
    """Process-local grammar compiler + matcher factory (one per engine process)."""

    def __init__(self, compiler: Any, tokenizer_info: Any, stop_token_ids: Sequence[int] = ()):
        self._compiler = compiler
        self.tokenizer_info = tokenizer_info
        self.stop_token_ids = tuple(sorted({int(t) for t in stop_token_ids}))
        self._grammars: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        *,
        vocab_size: int | None = None,
        stop_token_ids: Sequence[int] | None = None,
    ) -> "StructuredCompiler":
        """Build the compiler from the engine's already-loaded tokenizer.

        ``vocab_size`` is the MODEL's vocabulary (logits width), which some families pad
        beyond the tokenizer's own vocabulary; xgrammar needs it so the mask covers every
        logit column. Stop ids let the grammar terminate on the model's own EOS tokens.
        """
        if xgr is None:
            raise RuntimeError(
                "structured output requires the optional 'xgrammar' dependency "
                "(pip install xgrammar)"
            )
        stop_ids = sorted({int(t) for t in (stop_token_ids or [])})
        try:
            info = xgr.TokenizerInfo.from_huggingface(
                tokenizer,
                vocab_size=vocab_size,
                stop_token_ids=stop_ids or None,
            )
        except Exception:
            info = _tokenizer_info_from_vocab(tokenizer, vocab_size, stop_ids)
        return cls(xgr.GrammarCompiler(info), info, stop_token_ids=stop_ids)

    def new_state(self, spec: dict[str, Any]) -> StructuredState:
        """Compile (or fetch from cache) the spec's grammar and open a fresh matcher.

        ``terminate_without_stop_token`` is the contract this server promises: the request
        finishes the moment the document is complete, it never has to emit (and the client
        never receives) a trailing stop token. The scheduler's finish check reads
        ``StructuredState.terminated`` for that.
        """
        compiled = self._compiled(spec)
        if not self.stop_token_ids:
            raise RuntimeError(
                "structured output needs the model's stop token ids to close a completed "
                "document (see StructuredCompiler.from_tokenizer)"
            )
        return StructuredState(
            matcher=xgr.GrammarMatcher(compiled, terminate_without_stop_token=True),
            spec=spec,
            stop_token_ids=self.stop_token_ids,
        )

    def _compiled(self, spec: dict[str, Any]) -> Any:
        key = cache_key(spec)
        with self._lock:
            cached = self._grammars.get(key)
            if cached is not None:
                return cached
            if spec.get("type") == "json_object":
                compiled = self._compiler.compile_builtin_json_grammar()
            else:
                compiled = self._compiler.compile_json_schema(
                    spec["schema"], strict_mode=bool(spec.get("strict", False))
                )
            if len(self._grammars) >= MAX_CACHED_GRAMMARS:
                self._grammars.pop(next(iter(self._grammars)))
            self._grammars[key] = compiled
            return compiled


def _tokenizer_info_from_vocab(
    tokenizer: Any, vocab_size: int | None, stop_ids: list[int]
) -> Any:
    """Fallback for tokenizers ``from_huggingface`` cannot introspect.

    Builds the byte-level vocabulary in id order from ``get_vocab()``; the byte-level
    mapping ("Ġ" for a space) is exactly what ``VocabType.BYTE_LEVEL`` expects.
    """
    vocab = tokenizer.get_vocab()
    id_to_token = {int(i): t for t, i in vocab.items()}
    highest = max(id_to_token)
    encoded = [id_to_token.get(i, "") for i in range(highest + 1)]
    return xgr.TokenizerInfo(
        encoded,
        xgr.VocabType.BYTE_LEVEL,
        vocab_size=vocab_size or len(encoded),
        stop_token_ids=stop_ids or None,
    )


def states_of(reqs: Sequence[Any]) -> list[StructuredState | None]:
    """Row-aligned states for a batch's requests (None for plain rows)."""
    return [getattr(req, "structured_state", None) for req in reqs]


#: Set once, at the first backend fallback, so a device that cannot use xgrammar's native
#: kernel says so loudly instead of paying the fallback cost silently every step.
_FALLBACK_WARNED = False


def _allow_token(bitmask: torch.Tensor, row: int, token_id: int) -> None:
    """Set one token's bit in a CPU int32 bitmask (xgrammar's layout, little-endian word)."""
    word, bit = divmod(int(token_id), 32)
    value = 1 << bit
    if value > 0x7FFFFFFF:  # bit 31 does not fit a signed int32
        value -= 0x100000000
    bitmask[row, word] |= value


def mask_logits(logits: torch.Tensor, states: Sequence[StructuredState | None]) -> torch.Tensor:
    """Apply grammar masks to the constrained rows of a batch's logits.

    Returns ``logits`` unchanged when no row is constrained (the common case), otherwise a
    clone whose constrained rows have every token the grammar forbids set to -inf.

    A row whose grammar has already completed (possible under overlap scheduling, which may
    have launched one more step before the drain could finish the request) is masked down to
    the model's stop tokens: the extra step emits a stop token that the detokenizer drops,
    instead of a token that would be appended to a JSON document the client already holds.
    """
    global _FALLBACK_WARNED

    if xgr is None:  # defensive: the API refuses structured requests without xgrammar
        raise RuntimeError("structured output requires the optional 'xgrammar' dependency")
    rows = [i for i, state in enumerate(states) if state is not None]
    if not rows:
        return logits

    masked = logits.clone()
    bitmask = xgr.allocate_token_bitmask(masked.shape[0], masked.shape[1])
    xgr.reset_token_bitmask(bitmask)
    for row in rows:
        state = states[row]
        if state.terminated:
            # reset_token_bitmask fills all ones (allow-all); the stop-only mask starts empty.
            bitmask[row].fill_(0)
            for token_id in state.stop_token_ids:
                _allow_token(bitmask, row, token_id)
        else:
            state.matcher.fill_next_token_bitmask(bitmask, row)
    # xgrammar's apply requires the bitmask beside the logits; the fill itself runs on the
    # CPU buffer, which is what xgrammar writes to.
    bitmask = bitmask.to(masked.device)
    try:
        xgr.apply_token_bitmask_inplace(masked, bitmask, indices=rows)
    except Exception:  # noqa: BLE001 -- backend quirk: retry with the torch kernel
        if not _FALLBACK_WARNED:
            _FALLBACK_WARNED = True
            logger.warning(
                "xgrammar's native masking backend failed on this device; falling back to "
                "torch_native for this and later steps",
                exc_info=True,
            )
        xgr.apply_token_bitmask_inplace(masked, bitmask, indices=rows, backend="torch_native")
    return masked


def advance_states(reqs: Sequence[Any], next_tokens: torch.Tensor) -> None:
    """Advance the grammar of every constrained row with the token just sampled.

    Runs in ``Engine.forward_batch`` right after sampling, NOT in the scheduler's drain:
    overlap scheduling prepares and launches batch N before draining N-1, so an advance in
    the drain would leave the next batch's mask one token stale -- the matcher would then
    reject tokens the mask had allowed. The host read below is the price of that ordering
    and is paid only by batches that carry a constrained row.
    """
    rows = [
        (i, req)
        for i, req in enumerate(reqs)
        if getattr(req, "structured_state", None) is not None
    ]
    if not rows:
        return
    tokens = next_tokens.tolist()  # one sync; structured batches only
    for i, req in rows:
        state = req.structured_state
        if state.terminated:
            continue
        if not state.accept(tokens[i]):
            raise RuntimeError(
                f"structured grammar rejected sampled token {tokens[i]} for uid {req.uid}: "
                "the token mask and the sampler disagree"
            )
