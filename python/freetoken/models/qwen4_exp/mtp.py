"""Qwen3.8-Flash-Next MTP draft head (``mtp.*``).

One draft block per MTP layer, fed from the TARGET's hidden state rather than from a
re-run prompt (the DSpark structure from upstream #69): the target's final 4-stream
hyper-connection residual and the draft tokens' embeddings are normalized, projected
to draft width separately and summed, then run through full-attention decoder layers
whose context KV is derived from that projection alone. The draft never runs GDN or
PLE -- its context is a few projected tokens, not the prompt -- and it shares the
target's ``embed_tokens`` and ``lm_head`` (it owns no ``mtp.embed_tokens`` /
``mtp.lm_head`` keys).

Checkpoint ground truth (RadixArk/Qwen3.8-Flash-Next-NVFP4, read-only ssh): the
projection is split -- ``mtp.fc_embedding`` [H, H] over the normalized draft embeds
plus ``mtp.fc_hidden`` [H, H] over each normalized stream of the 4-stream target
residual (``mtp.pre_fc_norm_hidden`` is [4H], RMS per stream then the full-width
gamma; the block applies ``[W_e | W_h]`` per stream, so no stream pooling). There is
no ``mtp.fc`` key on this checkpoint; the vLLM single-``fc`` shape this module first
assumed does not load here. Likewise the routed experts are dense BF16 stacked
tensors (``gate_up_proj`` [E, 2I, H], ``down_proj`` [E, H, I], no ``_scale_inv``
companions) served from bf16 offload banks, not block-fp8.

Structure otherwise follows vLLM's ``Qwen3NextMultiTokenPredictor`` adapted to the
HC stack: the layers are :class:`Qwen4ExpDecoderLayer` forced to full attention, and
the trailing ``hyper_connection_mixer`` (no combine) replaces its final norm. The
checkpoint names are the layer's own (``mtp.layers.{i}.self_attn.qkv_proj.weight``
fuses in-namespace through the same loader fusions as the target), so
``iter_weights(include_mtp=True)`` feeds ``load_state_dict`` directly.

Two integration notes for the verify driver (419.7):

* Draft layers own MoE bank ids ``num_moe_layers + i`` (matching
  ``iter_mtp_expert_pieces``) and QSA slot ids of the same numbers. The QSA backend is
  sized for the target's layers, so the driver must provide the draft layers' KV pages
  and slots; the head itself only runs the math.
* The engine loads ``mtp.*`` dense weights only when the head exists
  (``include_mtp=model.mtp is not None``); until then the loader defaults stay inert
  and ``load_state_dict`` is strict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearReplicated, OPList

from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .model import Qwen4ExpDecoderLayer

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


class Qwen4ExpMTPHead(BaseOP):
    """The speculative draft stack: ``(draft embeds, target hidden) -> draft hidden``.

    ``num_layers`` is the MTP depth (1 for the shipping checkpoint's ``mtp.layers.0``).
    Draft layer ``i`` is a full-attention decoder layer with MoE bank id
    ``num_moe_layers + i``; the returned hidden feeds the target's shared ``lm_head``.
    """

    def __init__(self, config: ModelConfig, num_layers: int, *, prefix: str = "mtp") -> None:
        assert num_layers >= 1, num_layers
        args = config.qwen4_args
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        # Plain per-hidden RMSNorms over the fc inputs, zero-centered like
        # every other qwen4_exp norm. The embedding norm covers one stream [H];
        # the hidden norm normalizes EACH stream on its own statistic (group_size
        # = H, the reference's reshape [4H] -> [H, hc]) and then scales by the
        # full [hc*H] gamma -- not one stat over the whole 4-stream vector.
        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(
            args.hidden_size, config.rms_norm_eps, 1)
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            args.hc_count * args.hidden_size, config.rms_norm_eps, args.hc_count)
        # Split projection back to draft width (bf16 dense like the rest of the
        # head): fc_embedding over the normalized draft embeds plus fc_hidden over
        # the stream-mean of the normalized target residual, summed. A decomposed
        # concat+project (cross terms trained to zero); the split is forced by the
        # checkpoint, which ships no mtp.fc key.
        self.fc_embedding = LinearReplicated(
            args.hidden_size, args.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.fc_embedding",
        )
        self.fc_hidden = LinearReplicated(
            args.hidden_size, args.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.fc_hidden",
        )
        self.layers = OPList([
            Qwen4ExpDecoderLayer(
                config, config.num_moe_layers + i,
                prefix=f"{prefix}.layers.{i}", force_full_attention=True,
            )
            for i in range(num_layers)
        ])
        self._num_moe_layers = config.num_moe_layers
        self.hyper_connection_mixer = GatedResidual(
            config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer")
        # The pre-mixer 4-stream residual of the last draft evaluation, for the
        # driver's hidden chaining (each draft position conditions on the anchor
        # residual or the previous draft residual, never on post-mixer hidden).
        self.last_draft_residual: torch.Tensor | None = None

    @property
    def num_layers(self) -> int:
        return len(self.layers.op_list)

    def draft_layer_ids(self) -> list[int]:
        """MoE-bank (and QSA-slot) ids of the draft layers: past the target's MoE layers,
        matching ``iter_mtp_expert_pieces``."""
        return [self._num_moe_layers + i for i in range(self.num_layers)]

    def project(self, inputs_embeds: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        """The draft block's entry residual (``[T, hc*H]``) from embeds + 4-stream hidden.

        ``inputs_embeds`` are the draft tokens embedded with the TARGET's table
        (``[T, H]``); ``target_hidden`` is the target's final pre-mixer residual
        (``[T, hc*H]``). The projection is PER STREAM, mirroring the reference's
        ``concat([e_norm, h_norm], dim=stream) @ [W_e | W_h]``: every stream keeps its
        own normalized residual through fc_hidden (pooling the streams first would
        discard the hyper-connection signal the block was trained on). The same
        normalized embedding feeds all streams.
        """
        t = inputs_embeds.shape[0]
        hc, h = self.hc_count, self.hidden_size
        assert target_hidden.shape == (t, hc * h), (
            tuple(target_hidden.shape), t, hc, h)
        embeds = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(inputs_embeds))
        normed = self.pre_fc_norm_hidden.forward(target_hidden).view(t, hc, h)
        streamed = self.fc_hidden.forward(normed.reshape(t * hc, h)).view(t, hc, h)
        return (streamed + embeds.unsqueeze(1)).reshape(t, hc * h)

    def forward(
        self, inputs_embeds: torch.Tensor, target_hidden: torch.Tensor, batch: Batch,
    ) -> torch.Tensor:
        """Draft hidden states ``[T, hidden]`` for the shared lm_head.

        ``inputs_embeds`` are the draft tokens embedded with the TARGET's embedding
        table, ``target_hidden`` is the target's final 4-stream residual at the same
        ``T`` positions; ``batch`` positions/page table describe those positions for the
        draft layers' attention. Also records the pre-mixer draft residual
        (:attr:`last_draft_residual`) for the driver's hidden chaining.
        """
        residual = self.project(inputs_embeds, target_hidden)
        for layer in self.layers.op_list:
            residual = layer.forward(residual, batch)
        self.last_draft_residual = residual
        return self.hyper_connection_mixer.mix(residual)[0]


__all__ = ["Qwen4ExpMTPHead"]
