"""The MTP draft head: structure, loader contract, and CPU-checkable math.

The head reuses the target's decoder layer forced to full attention, so its dense
``mtp.*`` keys must be exactly what ``iter_weights(include_mtp=True)`` yields (fusions
in-namespace included) while the stacked experts stay in the offload banks. The full
forward needs a GPU (Triton norms/MoE), so it is gated like the skeleton's decoder test;
everything else runs here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
from freetoken.models.qwen4_exp.mtp import Qwen4ExpMTPHead
from freetoken.models.qwen4_exp.weight import iter_weights

from .common import requires_cuda, toy_hf_config


@pytest.fixture(scope="module", autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _config(**overrides):
    return parse_config(toy_hf_config(4, **overrides))


def _head(num_layers: int = 1) -> tuple:
    config = _config()
    return config, Qwen4ExpMTPHead(config, num_layers)


# --------------------------------------------------------------------------------------
# Wiring: absent by default, explicit when asked
# --------------------------------------------------------------------------------------


def test_default_model_builds_no_mtp_submodule():
    model = Qwen4ExpForCausalLM(_config())
    assert model.mtp is None
    assert not [k for k in model.state_dict() if k.startswith("mtp.")]


def test_explicit_count_builds_the_head():
    model = Qwen4ExpForCausalLM(_config(), mtp_num_layers=2)
    assert model.mtp.num_layers == 2
    assert model.mtp.draft_layer_ids() == [model._config.num_moe_layers + i for i in (0, 1)]
    assert any(k.startswith("mtp.") for k in model.state_dict())


def test_config_carries_the_opt_in_depth_default_off():
    assert _config().qwen4_args.mtp_num_layers == 0
    assert _config(num_nextn_predict_layers=1).qwen4_args.mtp_num_layers == 1
    model = Qwen4ExpForCausalLM(_config(num_nextn_predict_layers=1))
    assert model.mtp is not None and model.mtp.num_layers == 1


def test_target_residual_side_channel_starts_empty():
    """The engine's anchor source: no hidden is exposed before the first forward."""
    model = Qwen4ExpForCausalLM(_config(num_nextn_predict_layers=1))
    assert model.last_target_hidden is None
    assert model.model.last_target_residual is None


def test_draft_layers_are_full_attention_banked_past_the_target():
    config, head = _head()
    target_layer = config.num_moe_layers  # == num_layers here (every layer is MoE)
    assert head.draft_layer_ids() == [target_layer]
    for i, layer in enumerate(head.layers.op_list):
        assert not layer._is_linear
        assert layer._layer_id == target_layer + i
        assert layer.mlp is not None and layer.ple is None


# --------------------------------------------------------------------------------------
# Loader contract: iter_weights(include_mtp=True) feeds load_state_dict directly
# --------------------------------------------------------------------------------------


def _raw_parts(name: str, tensor: torch.Tensor, config):
    """Module key -> (raw checkpoint key, raw tensor) parts, inverting the loader fusions."""
    gen = torch.Generator().manual_seed(1234)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=tensor.dtype)

    if name.endswith(".self_attn.qkv_proj.weight"):
        qo2, kv, h = 2 * config.num_qo_heads * config.head_dim, config.num_kv_heads * config.head_dim, config.hidden_size
        assert tuple(tensor.shape) == (qo2 + 2 * kv, h)
        base = name[: -len(".qkv_proj.weight")]
        parts = tensor.split([qo2, kv, kv])
        return [(f"{base}.{p}.weight", t) for p, t in zip(("q_proj", "k_proj", "v_proj"), parts)]
    if name.endswith(".input_mix_weight_down_block_inject.weight"):
        lr, hc, h = config.qwen4_args.hc_lowrank, config.qwen4_args.hc_count, config.qwen4_args.ple_state_width
        pad = (-(lr + hc)) % 16
        assert tuple(tensor.shape) == (lr + hc + pad, h)
        base = name[: -len(".input_mix_weight_down_block_inject.weight")]
        down, inject, _pad = tensor.split([lr, hc, pad])
        return [(f"{base}.input_mix_weight_down.weight", down),
                (f"{base}.block_inject_weight.weight", inject)]
    if name.endswith(".mlp.shared_expert.gate_up_proj.weight"):
        inter, h = config.shared_expert_intermediate_size, config.hidden_size
        assert tuple(tensor.shape) == (2 * inter, h)
        base = name[: -len(".gate_up_proj.weight")]
        gate, up = tensor.split([inter, inter])
        return [(f"{base}.gate_proj.weight", gate), (f"{base}.up_proj.weight", up)]
    return [(name, rand(*tensor.shape))]


def _mtp_raw_fixture(head: Qwen4ExpMTPHead, config) -> dict[str, torch.Tensor]:
    """Raw (pre-fusion) checkpoint tensors covering every dense mtp key the head owns."""
    raw: dict[str, torch.Tensor] = {}
    for name, tensor in head.state_dict().items():
        if ".mlp.experts." in name:
            continue  # bank-backed, never in the dense dict (asserted separately)
        for raw_name, raw_tensor in _raw_parts("mtp." + name, tensor, config):
            raw[raw_name] = raw_tensor
    # The real stacked experts (skipped unread by the renamer): shapes are irrelevant here.
    for role in ("gate_up_proj", "down_proj"):
        raw[f"mtp.layers.0.mlp.experts.{role}"] = torch.zeros(1)
    return raw


def test_loader_yields_exactly_the_head_dense_keys(tmp_path):
    config, head = _head()
    save_file(_mtp_raw_fixture(head, config), str(tmp_path / "model.safetensors"))
    yielded = dict(iter_weights(
        str(tmp_path), torch.device("cpu"),
        include_moe_experts=True, include_non_moe=True, include_mtp=True))
    mtp_yielded = {k[4:]: v for k, v in yielded.items() if k.startswith("mtp.")}
    expected = {k for k in head.state_dict() if ".mlp.experts." not in k}
    assert set(mtp_yielded) == expected
    # strict load: the yielded dense dict plus correctly-shaped expert fillers leaves nothing
    state = dict(mtp_yielded)
    for name, tensor in head.state_dict().items():
        if ".mlp.experts." in name:
            state[name] = torch.zeros_like(tensor)
    head.load_state_dict(state)
    assert state == {}


def test_stacked_experts_never_enter_the_dense_dict(tmp_path):
    config, head = _head()
    save_file(_mtp_raw_fixture(head, config), str(tmp_path / "model.safetensors"))
    yielded = dict(iter_weights(
        str(tmp_path), torch.device("cpu"),
        include_moe_experts=True, include_non_moe=True, include_mtp=True))
    assert not [k for k in yielded if ".mlp.experts." in k]


def test_loader_defaults_stay_inert(tmp_path):
    config, head = _head()
    save_file(_mtp_raw_fixture(head, config), str(tmp_path / "model.safetensors"))
    yielded = dict(iter_weights(
        str(tmp_path), torch.device("cpu"),
        include_moe_experts=True, include_non_moe=True))
    assert not [k for k in yielded if k.startswith("mtp.")]


# --------------------------------------------------------------------------------------
# CPU-checkable math: the projection path and the trailing mixer
# --------------------------------------------------------------------------------------


def test_fc_projection_path_matches_manual_torch():
    """Split projection, PER STREAM: the reference applies ``[W_e | W_h]`` to
    ``concat([e_norm, h_norm], dim=stream)``, so each stream keeps its own normalized
    residual (pooling first would discard the hyper-connection signal)."""
    config, head = _head()
    torch.manual_seed(0)
    for tensor in head.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, 0.05)
    T, H, hc = 4, config.hidden_size, config.qwen4_args.hc_count
    embeds = torch.randn(T, H) * 0.1
    hidden = torch.randn(T, hc * H) * 0.1

    def rms(x, w, eps):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
                * (1.0 + w.float())).to(x.dtype)

    def rms_grouped(x, w, eps, groups):
        # per-stream stats, then the full-width gamma (HF RMSNorm(group_size=H))
        xf = x.float().unflatten(-1, (groups, -1))
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (out.flatten(-2) * (1.0 + w.float())).to(x.dtype)

    want_e = rms(embeds, head.pre_fc_norm_embedding.weight, config.rms_norm_eps)
    assert tuple(head.pre_fc_norm_hidden.weight.shape) == (hc * H,)
    want_h = rms_grouped(hidden, head.pre_fc_norm_hidden.weight, config.rms_norm_eps, hc)
    want = torch.empty(T, hc, H)
    for s in range(hc):
        want[:, s] = (F.linear(want_e, head.fc_embedding.weight)
                      + F.linear(want_h.view(T, hc, H)[:, s], head.fc_hidden.weight))
    assert torch.equal(head.project(embeds, hidden), want.view(T, hc * H))
    # determinism: the projection is a pure function of its inputs
    assert torch.equal(head.project(embeds, hidden), want.view(T, hc * H))


def test_project_rejects_a_single_stream_hidden():
    """The head consumes the pre-mixer 4-stream residual, never post-mixer hidden."""
    config, head = _head()
    T, H = 2, config.hidden_size
    with pytest.raises(AssertionError):
        head.project(torch.randn(T, H), torch.randn(T, H))


def test_head_state_dict_matches_the_shipping_checkpoint_layout():
    """Pin the ssh ground truth (RadixArk/Qwen3.8-Flash-Next-NVFP4, 31 mtp.* keys):
    split fc_embedding/fc_hidden (no mtp.fc), a 4H-wide pre_fc_norm_hidden, dense-bf16 stacked experts (no _scale_inv).
    Post-load names: the loader fuses q|k|v, the shared gate|up and each HC
    down|inject pair in-namespace. Uses the offload toy model (production strategy):
    bank-backed experts own no dense rows."""
    offload_config, model = _offload_toy_model()
    config, head = offload_config, model.mtp
    H = config.hidden_size
    L0 = "mtp.layers.0"
    hc = ("hc_norm.weight", "input_mix_weight_down_block_inject.weight",
          "input_mix_weight_up.weight")
    got = set(head.state_dict())
    want = {
        "mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
        "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        f"{L0}.self_attn.qkv_proj.weight", f"{L0}.self_attn.o_proj.weight",
        f"{L0}.self_attn.q_norm.weight", f"{L0}.self_attn.k_norm.weight",
        f"{L0}.self_attn.indexer.index_qk_proj.weight",
        f"{L0}.self_attn.indexer.q_layernorm.weight",
        f"{L0}.self_attn.indexer.k_layernorm.weight",
        f"{L0}.mlp.gate.weight", f"{L0}.mlp.shared_expert.gate_up_proj.weight",
        f"{L0}.mlp.shared_expert.down_proj.weight", f"{L0}.mlp.shared_expert_gate.weight",
    }
    for block in ("attn_hyper_connection", "mlp_hyper_connection"):
        want |= {f"{L0}.{block}.{leaf}" for leaf in hc}
    # Bank-backed experts own no state-dict rows (their stacked checkpoint tensors
    # flow through iter_mtp_expert_pieces, never the dense dict). Keys are
    # head-relative here (the parent model adds the mtp. prefix).
    want = {k[len("mtp."):] for k in want}
    assert got == want, (sorted(got ^ want))
    assert head.pre_fc_norm_hidden.weight.numel() == 4 * H
    assert head.fc_embedding.weight.shape == (H, H)
    assert head.fc_hidden.weight.shape == (H, H)


def test_top_mixer_matches_the_hf_formula():
    """The trailing mixer is GatedResidual without combine (mix only), checked against the
    HF Qwen4ExpTextGatedResidual math like the skeleton's target-mixer test."""
    config, head = _head()
    torch.manual_seed(1)
    for tensor in head.hyper_connection_mixer.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, 0.05)
    hc, H = config.qwen4_args.hc_count, config.hidden_size
    R = torch.randn(5, hc * H) * 0.1
    mixer = head.hyper_connection_mixer

    xf = R.float().reshape(*R.shape[:-1], hc, -1)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
    xn = (xf.flatten(-2) * (1.0 + mixer.hc_norm.weight.float())).to(R.dtype)
    lora = F.linear(xn, mixer.input_mix_weight_down.weight)
    gate = torch.sigmoid(F.linear(F.silu(lora / hc).to(R.dtype),
                                  mixer.input_mix_weight_up.weight))
    want = (gate.unflatten(-1, (hc, H)) * xn.unflatten(-1, (hc, H))).mean(-2)
    got, s = mixer.mix(R)
    assert s is None
    assert torch.allclose(got, want, atol=1e-5)


# --------------------------------------------------------------------------------------
# Draft banks: packed through the draft layers' own method, attached to a dedicated
# bf16 draft cache (one cache holds one bank schema, so the NVFP4 target cache
# cannot serve them). CPU-runnable; the GPU box only changes the device.
# --------------------------------------------------------------------------------------


def _offload_toy_model(num_mtp_layers: int = 1):
    from dataclasses import replace

    from freetoken.layers.quantization.configs import NoQuantConfig

    # A bare quant config (like production's checkpoint-derived one) binds every
    # MoE layer -- target and draft -- to the unquantized bf16 method.
    config = replace(_config(), moe_strategy="offload", quant=NoQuantConfig())
    return config, Qwen4ExpForCausalLM(config, mtp_num_layers=num_mtp_layers)


def _write_toy_mtp_ckpt(folder: str, config) -> dict:
    E, I, H = config.num_experts, config.moe_intermediate_size, config.hidden_size
    base = "mtp.layers.0.mlp.experts"
    tensors = {
        f"{base}.gate_up_proj": torch.randn(E, 2 * I, H, dtype=torch.bfloat16),
        f"{base}.down_proj": torch.randn(E, H, I, dtype=torch.bfloat16),
    }
    import os

    save_file(tensors, os.path.join(folder, "model.safetensors"))
    return tensors


def test_build_mtp_draft_banks_packs_through_the_draft_layers_method(tmp_path):
    from freetoken.engine.mtp import build_mtp_draft_banks, draft_moe_layers

    config, model = _offload_toy_model(1)
    raw = _write_toy_mtp_ckpt(str(tmp_path), config)
    layers = draft_moe_layers(model)
    assert len(layers) == 1
    assert layers[0].quant_method is not None  # unquantized bf16 on the test config

    banks = build_mtp_draft_banks(
        model, str(tmp_path), config, 1, device=torch.device("cpu"))

    assert banks.quant_format == "bf16"
    assert set(banks.sources) == {"gate_up", "down"}
    assert len(banks.sources["gate_up"]) == 1  # local draft rows, not global ids
    assert torch.equal(banks.sources["gate_up"][0], raw["mtp.layers.0.mlp.experts.gate_up_proj"])
    assert torch.equal(banks.sources["down"][0], raw["mtp.layers.0.mlp.experts.down_proj"])


def test_attach_mtp_draft_cache_rebinds_to_local_rows(tmp_path):
    from freetoken.engine.mtp import (
        attach_mtp_draft_cache,
        build_mtp_draft_banks,
        draft_moe_layers,
    )
    from freetoken.moe.offload_cache import OffloadMoeCache

    config, model = _offload_toy_model(1)
    _write_toy_mtp_ckpt(str(tmp_path), config)
    layers = draft_moe_layers(model)
    assert layers[0].layer_id == config.num_moe_layers  # global id at construction
    banks = build_mtp_draft_banks(
        model, str(tmp_path), config, 1, device=torch.device("cpu"))
    method = layers[0].quant_method
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=config.num_experts,
        cache_size=config.num_experts,
        device=torch.device("cpu"),
        prefill_overlap=False,
        quant_format=banks.quant_format,
        decode_target="gpu",
        layout=method.layout(),
        max_slots=method.slot_limit(),
    )
    cache.set_bank_sources(banks.sources)

    attached = attach_mtp_draft_cache(model, cache)

    assert attached == layers
    assert layers[0].layer_id == 0  # local draft row
    assert layers[0].offload_cache is cache


def test_shared_offload_method_skips_pre_attached_draft_layers():
    from types import SimpleNamespace

    from freetoken.engine.engine import shared_offload_method
    from freetoken.engine.mtp import draft_moe_layers

    config, model = _offload_toy_model(1)
    target_method = shared_offload_method(model)  # all-unquantized test config agrees
    assert target_method is not None
    (draft,) = draft_moe_layers(model)
    # A foreign-format draft method disagrees with the target while unattached ...
    draft.quant_method = SimpleNamespace(kind="FP8_BLOCK", kernel=SimpleNamespace(name="x"))
    with pytest.raises(ValueError, match="disagree"):
        shared_offload_method(model)
    # ... but is skipped once the draft layer is bound to its own cache.
    draft.offload_cache = SimpleNamespace()
    assert shared_offload_method(model) is target_method


# --------------------------------------------------------------------------------------
# Full forward (needs a GPU for the Triton norms/MoE -- skipped here, runs on GPU CI)
# --------------------------------------------------------------------------------------


@requires_cuda
def test_draft_forward_runs():
    """Draft embeds + target hidden through the whole head with dummy weights."""
    from types import SimpleNamespace

    from freetoken.core import Context, set_global_ctx
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference
    from freetoken.utils.torch_utils import torch_dtype
    import freetoken.core as core

    torch.manual_seed(8)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = _config()
    args = config.qwen4_args
    with torch.device(device), torch_dtype(dtype):
        head = Qwen4ExpMTPHead(config, 1)
    gen = torch.Generator(device=device).manual_seed(9)
    for tensor in head.state_dict().values():
        if tensor.is_floating_point():
            tensor.normal_(0.0, 0.05, generator=gen)
    T = 3
    embeds = torch.randn(T, args.hidden_size, generator=gen, device=device, dtype=dtype) * 0.05
    hidden = torch.randn(T, args.hc_count * args.hidden_size, generator=gen, device=device, dtype=dtype) * 0.05
    backend = TorchDenseQSAReference(config, 4, 32, device, dtype)
    core._GLOBAL_CTX = None
    ctx = Context(page_size=64)
    ctx.attn_backend = backend
    set_global_ctx(ctx)
    req = SimpleNamespace(extend_len=T, cached_len=0, table_idx=1)
    batch = SimpleNamespace(
        padded_reqs=[req], reqs=[req], is_prefill=True, is_decode=False,
        input_ids=torch.arange(T, device=device),
        positions=torch.arange(T, device=device),
        attn_metadata=None, fla_metadata=None, linear_table_idx=None,
    )
    with ctx.forward_batch(batch):
        out = head.forward(embeds, hidden, batch)
    assert out.shape == (T, args.hidden_size)
    assert torch.isfinite(out.float()).all()


# --------------------------------------------------------------------------------------
# Real-checkpoint key cover: the RadixArk NVFP4 index ships 31 mtp.* keys (2026-09-09).
# The head-derived synthetic fixtures above are circular by construction (raws built
# from the head's own keys), so this pins both sides against the checkpoint's list:
# every dense raw key must surface (fused in-namespace) and the head must own every
# surfaced key -- otherwise strict load_state_dict fails on the GPU box.
# --------------------------------------------------------------------------------------

_REAL_MTP_KEYS = [
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.experts.down_proj",
    "mtp.layers.0.mlp.experts.gate_up_proj",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]

# The reader's fused output for the list above: q|k|v -> qkv_proj, shared gate|up ->
# gate_up_proj, per-layer-HC down|inject -> input_mix_weight_down_block_inject.
_EXPECTED_MTP_YIELDED = {
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down_block_inject.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.qkv_proj.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}


def _real_key_raws() -> dict[str, torch.Tensor]:
    """Tiny tensors under the checkpoint's real key names (shapes only need to let
    the three fusions complete; key coverage, not values, is under test)."""
    H = 8
    raw: dict[str, torch.Tensor] = {}
    for key in _REAL_MTP_KEYS:
        if key.endswith("experts.gate_up_proj"):
            raw[key] = torch.zeros(1)
        elif key.endswith("experts.down_proj"):
            raw[key] = torch.zeros(1)
        elif key.endswith(".weight"):
            rows = 6 if "q_proj" in key else 4
            raw[key] = torch.randn(rows, H, dtype=torch.bfloat16)
        else:  # pragma: no cover - every real key ends in .weight or is an expert stack
            raise AssertionError(f"unhandled real key shape: {key}")
    return raw


def test_real_checkpoint_keys_all_surface(tmp_path):
    save_file(_real_key_raws(), str(tmp_path / "model.safetensors"))
    yielded = dict(iter_weights(
        str(tmp_path), torch.device("cpu"),
        include_moe_experts=True, include_non_moe=True, include_mtp=True))
    assert set(yielded) == _EXPECTED_MTP_YIELDED


def test_head_state_dict_matches_the_real_checkpoint_cover():
    """The head must own exactly the keys the checkpoint serves (fused): no missing
    module (strict-load KeyError on the GPU box), no extra module (unexpected-key)."""
    _config, head = _head()
    # head-relative keys (the parent model adds the mtp. prefix)
    assert {"mtp." + k for k in head.state_dict() if ".mlp.experts." not in k} == _EXPECTED_MTP_YIELDED


def test_lm_head_all_rows_keeps_every_prefill_row():
    """The draft forward needs every row's distribution; the head's default prefill
    slicing (last row per request) would collapse a T-row draft prefix to one row
    (live IndexError signature: propose_drafts got [1, V] for a 2-token prefix)."""
    from types import SimpleNamespace

    import freetoken.core as core
    from freetoken.core import Context, set_global_ctx
    from freetoken.layers.embedding import ParallelLMHead, VocabParallelEmbedding

    from .common import fresh_ctx

    V, H, T = 32, 8, 3
    embed = VocabParallelEmbedding(num_embeddings=V, embedding_dim=H)
    head = ParallelLMHead(
        num_embeddings=V, embedding_dim=H,
        tie_word_embeddings=True, tied_embedding=embed)
    # torch.empty weights may hold recycled NaN bit patterns (NaN != NaN breaks
    # torch.equal); fill deterministically so the comparison is bitwise exact.
    torch.manual_seed(0)
    embed.weight.normal_(0.0, 0.05)
    x = torch.randn(T, H) * 0.1
    last = torch.tensor([T - 1])
    batch = SimpleNamespace(
        size=1, is_prefill=True,
        attn_metadata=SimpleNamespace(get_last_indices=lambda bs: last[:bs]))
    fresh_ctx()
    core._GLOBAL_CTX._batch = batch
    assert head.forward(x).shape == (1, V)
    got = head.forward(x, all_rows=True)
    assert got.shape == (T, V)
    assert torch.equal(got, torch.nn.functional.linear(x, embed.weight))
