from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_embeds: torch.Tensor | None = None
    # per-image processor outputs, in prompt order (None for text-only requests)
    mm_items: list | None = None
    # precomputed [3, len(input_ids)] mrope positions and decode delta; None for text-only
    # requests and 1-D rope models
    mrope_positions_full: torch.Tensor | None = None
    mrope_delta: int = 0
    # Grammar-constrained decoding state (engine/structured.StructuredState). Built once at
    # admission from ``sampling_params.structured_output`` and handed to the request's Req --
    # chunked prefill rows carry no state (their samples are discarded), so the grammar only
    # ever advances on tokens the client actually receives.
    structured_state: object | None = None
    # Teacher-forced scoring (internal /v1/score): score instead of generate; ``score_chunk``
    # caps the rows per prefill forward (0 = the normal prefill budget). Kept on the pending
    # record so every chunk continuation inherits it.
    score_only: bool = False
    score_chunk: int = 0

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
