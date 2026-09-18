"""TP=2 dense-weight sharding and the rank-sharded NVFP4 expert banks for qwen4_exp.

The ambient TP info is swapped at the module level (restored at teardown) so these
tests coexist with the TP=1 suites: iter_weights and the TP-aware layers read the
same ``get_tp_info()`` global. The gate is end-to-end: the rank-0 shard dict produced
by ``iter_weights`` under tp (0, 2) must satisfy ``load_state_dict`` of the model the
layers actually declare (built on the meta device), and per-tensor reconstruction must
hold: the two ranks' shards re-cat to the TP=1 fused tensor.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from .common import hf_config

# geometry (all divisible by tp=2 with head alignment)
H = 32  # hidden_size
HC = 4  # hc_count
LR = 320  # hc_lowrank (real; keeps the merged HC pad at 12 rows)
HCH = HC * H
KH, VH, HD = 2, 6, 8  # GDN key / value heads, head dim
QH, KVH, AHD = 4, 2, 64  # QSA q / kv heads, head dim (rope needs head_size in {64,128,256,512})
IHD = 16  # indexer head dim (strict config: must be >= head_dim * partial_rotary_factor)
E, I, SI = 8, 32, 32  # routed experts, moe intermediate, shared-expert intermediate
VOCAB = 12  # even, so both vocab shards are 6 rows


def _bf16(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.bfloat16)


def _text_cfg(**over) -> SimpleNamespace:
    base = dict(
        num_layers=2, head_dim=AHD, num_q=QH, num_kv=KVH, index_head_dim=IHD,
        index_heads=4, budget=16, ratio=4, hidden=H, max_position=4096,
        rope_theta=10000.0,
    )
    base.update(over)
    cfg = hf_config(**base)
    t = cfg.text_config
    t.layer_types = ["linear_attention", "qwen_sparse_attention"]  # layer 0 GDN+PLE, layer 1 QSA
    t.num_experts = E
    t.num_experts_per_tok = 2
    t.moe_intermediate_size = I
    t.shared_expert_intermediate_size = SI
    t.linear_num_key_heads = KH
    t.linear_num_value_heads = VH
    t.linear_key_head_dim = HD
    t.linear_value_head_dim = HD
    t.hc_count = HC
    t.hc_lowrank = LR
    t.vocab_size = VOCAB
    t.ple_layer_ids = [1]  # HF one-based -> decoder layer 0 (linear)
    return cfg


def _to_jsonable(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _to_jsonable(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _hc_weights(prefix: str, inject: bool) -> dict[str, torch.Tensor]:
    w = {
        f"{prefix}.hc_norm.weight": _bf16(HCH),
        f"{prefix}.input_mix_weight_down.weight": _bf16(LR, HCH),
        f"{prefix}.input_mix_weight_up.weight": _bf16(HCH, LR),
    }
    if inject:
        w[f"{prefix}.block_inject_weight.weight"] = _bf16(HC, HCH)
    return w


def _raw_checkpoint() -> dict[str, torch.Tensor]:
    lm = "model.language_model"
    raw: dict[str, torch.Tensor] = {
        f"{lm}.embed_tokens.weight": _bf16(VOCAB, H),
        "lm_head.weight": _bf16(VOCAB, H),
    }
    raw.update(_hc_weights(f"{lm}.hyper_connection_mixer", inject=False))
    for layer in (0, 1):
        raw.update(_hc_weights(f"{lm}.layers.{layer}.attn_hyper_connection", inject=True))
        raw.update(_hc_weights(f"{lm}.layers.{layer}.mlp_hyper_connection", inject=True))
        raw.update({
            f"{lm}.layers.{layer}.mlp.gate.weight": _bf16(E, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.gate_proj.weight": _bf16(SI, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.up_proj.weight": _bf16(SI, H),
            f"{lm}.layers.{layer}.mlp.shared_expert.down_proj.weight": _bf16(H, SI),
            f"{lm}.layers.{layer}.mlp.shared_expert_gate.weight": _bf16(1, H),
        })
    gdn = f"{lm}.layers.0.linear_attn"
    raw.update({
        f"{gdn}.in_proj_qkv.weight": _bf16(2 * KH * HD + VH * HD, H),
        f"{gdn}.in_proj_z.weight": _bf16(VH * HD, H),
        f"{gdn}.in_proj_b.weight": _bf16(VH, H),
        f"{gdn}.in_proj_a.weight": _bf16(VH, H),
        f"{gdn}.conv1d.weight": _bf16(2 * KH * HD + VH * HD, 1, 4),
        f"{gdn}.A_log": _bf16(VH),
        f"{gdn}.dt_bias": _bf16(VH),
        f"{gdn}.norm.weight": _bf16(HD),
        f"{gdn}.out_proj.weight": _bf16(H, VH * HD),
    })
    ple = f"{lm}.layers.0.ple"
    raw.update({
        f"{ple}.key_proj.weight": _bf16(HCH, 64),  # in = ple_embed_dim, not hidden
        f"{ple}.value_proj.weight": _bf16(H, 64),
        f"{ple}.norm_key.weight": _bf16(HCH),
        f"{ple}.norm_query.weight": _bf16(HCH),
        f"{ple}.norm_conv.weight": _bf16(HCH),
        f"{ple}.conv1d.weight": _bf16(HCH, 1, 4),
        f"{ple}.ple_embedding.layer_multipliers": torch.randint(1, 1 << 40, (3,)),
        f"{ple}.ple_embedding.ngram_heads_offsets": torch.arange(4),
        f"{ple}.ple_embedding.ngram_heads_vocab_sizes": torch.full((4,), 5),
    })
    attn = f"{lm}.layers.1.self_attn"
    raw.update({
        f"{attn}.q_proj.weight": _bf16(2 * QH * AHD, H),
        f"{attn}.k_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.v_proj.weight": _bf16(KVH * AHD, H),
        f"{attn}.o_proj.weight": _bf16(H, QH * AHD),
        f"{attn}.q_norm.weight": _bf16(AHD),
        f"{attn}.k_norm.weight": _bf16(AHD),
        f"{attn}.indexer.index_qk_proj.weight": _bf16(5 * IHD, H),
        f"{attn}.indexer.q_layernorm.weight": _bf16(IHD),
        f"{attn}.indexer.k_layernorm.weight": _bf16(IHD),
    })
    return raw


@pytest.fixture(scope="module")
def tp_global():
    """Swap the one-shot ambient TP global for this module, restore it afterwards."""
    from freetoken.distributed import info

    old = getattr(info, "_TP_INFO", None)
    yield info
    info._TP_INFO = old


def _set_tp(info, rank: int, size: int):
    info._TP_INFO = info.DistributedInfo(rank, size)


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    torch.manual_seed(7)
    folder = tmp_path_factory.mktemp("qwen4_exp_tp_ckpt")
    raw = _raw_checkpoint()
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    with open(folder / "config.json", "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(_text_cfg()), fh)
    return str(folder)


def _load_all(ckpt_dir: str):
    """iter_weights as TP=1, rank0 and rank1 (ambient global swapped per call)."""
    from freetoken.distributed import info
    from freetoken.models.qwen4_exp.weight import iter_weights

    out = {}
    for key, (rank, size) in {"full": (0, 1), 0: (0, 2), 1: (1, 2)}.items():
        _set_tp(info, rank, size)
        out[key] = {
            name: t.clone()
            for name, t in iter_weights(
                ckpt_dir, torch.device("cpu"),
                include_moe_experts=True, include_non_moe=True, include_vision=False,
            )
        }
    return out


@pytest.fixture(scope="module")
def loaded(ckpt_dir, tp_global):
    return _load_all(ckpt_dir)


def test_rank_shards_recat_to_full(loaded):
    """Head-aligned col splits: per segment, rank0 rows ++ rank1 rows == TP=1 rows."""
    full, r0, r1 = loaded["full"], loaded[0], loaded[1]
    assert set(full) == set(r0) == set(r1)

    # qkv: local rows [q 2heads*(2*AHD) | k 1 head | v 1 head]; full [q 512 | k 128 | v 128]
    n = "model.layers.1.self_attn.qkv_proj.weight"
    lq, lkv = QH * AHD, KVH * AHD // 2
    assert r0[n].shape == (lq + 2 * lkv, H)
    q_w, kv_w = 2 * QH * AHD, KVH * AHD
    for lo, hi, flo in ((0, lq, 0), (lq, lq + lkv, q_w), (lq + lkv, lq + 2 * lkv, q_w + kv_w)):
        assert torch.equal(torch.cat([r0[n][lo:hi], r1[n][lo:hi]], 0), full[n][flo:flo + 2 * (hi - lo)])
    # o_proj: input dim (columns) split
    n = "model.layers.1.self_attn.o_proj.weight"
    assert r0[n].shape == (H, QH * AHD // 2)
    assert torch.equal(torch.cat([r0[n], r1[n]], 1), full[n])
    # GDN in_proj: local [conv(8 q | 8 k | 24 v) | z 24 | b 3 | a 3]; full [q16|k16|v48|z48|b6|a6]
    n = "model.layers.0.linear_attn.in_proj.weight"
    assert r0[n].shape == (70, H)
    for lo, hi, flo in ((0, 8, 0), (8, 16, 16), (16, 40, 32), (40, 64, 80), (64, 67, 128), (67, 70, 134)):
        assert torch.equal(torch.cat([r0[n][lo:hi], r1[n][lo:hi]], 0), full[n][flo:flo + 2 * (hi - lo)])
    n = "model.layers.0.linear_attn.out_proj.weight"
    assert torch.equal(torch.cat([r0[n], r1[n]], 1), full[n])
    n = "model.layers.0.linear_attn.conv1d.weight"
    assert r0[n].shape == (40, 1, 4)
    for lo, hi, flo in ((0, 8, 0), (8, 16, 16), (16, 40, 32)):
        assert torch.equal(
            torch.cat([r0[n][lo:hi], r1[n][lo:hi]], 0),
            full[n][flo:flo + 2 * (hi - lo)].reshape(-1, 1, 4),
        )
    for n in ("model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.dt_bias"):
        assert r0[n].shape == (VH // 2,)
        assert torch.equal(torch.cat([r0[n], r1[n]], 0), full[n])
    # shared expert: gate|up segments split; down columns split. local rows [gate|up],
    # full rows [gate(32) | up(32)] -> each local half re-cats its own full block.
    n = "model.layers.0.mlp.shared_expert.gate_up_proj.weight"
    si = SI // 2
    assert r0[n].shape == (2 * si, H)
    assert torch.equal(torch.cat([r0[n][:si], r1[n][:si]], 0), full[n][:SI])
    assert torch.equal(torch.cat([r0[n][si:], r1[n][si:]], 0), full[n][SI:])
    n = "model.layers.0.mlp.shared_expert.down_proj.weight"
    assert torch.equal(torch.cat([r0[n], r1[n]], 1), full[n])
    # vocab parallel embedding / lm head
    for n in ("model.embed_tokens.weight", "lm_head.weight"):
        assert r0[n].shape == (VOCAB // 2, H)
        assert torch.equal(torch.cat([r0[n], r1[n]], 0), full[n])


def test_replicated_keys_unchanged(loaded):
    full, r0 = loaded["full"], loaded[0]
    for name, t in full.items():
        leaf = name
        if any(leaf.endswith(s) for s in (
            "mlp.gate.weight", "shared_expert_gate.weight",
            "hyper_connection.hc_norm.weight",
        )) or "_hyper_connection." in leaf or "hyper_connection_mixer" in leaf or ".ple." in leaf:
            assert torch.equal(r0[name], t), name
        if ".self_attn.q_norm" in leaf or ".self_attn.k_norm" in leaf or "indexer" in leaf:
            assert torch.equal(r0[name], t), name


def _load_model_state(loaded_rank, ckpt_dir, rank):
    import dataclasses

    from freetoken.distributed import info
    from freetoken.engine.engine import _materialize_loaded_weight_state_dict
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM
    from freetoken.utils.hf import cached_load_hf_config
    from freetoken.utils.torch_utils import torch_dtype

    _set_tp(info, rank, 2)
    try:
        cfg = dataclasses.replace(parse_config(cached_load_hf_config(ckpt_dir)), moe_strategy="offload")
        from freetoken.layers.rotary import set_rope_device

        set_rope_device(torch.device("cpu"))  # the engine does this before model init
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            model = Qwen4ExpForCausalLM(cfg)
        state = _materialize_loaded_weight_state_dict(
            model.state_dict(), dict(loaded_rank).items(), device=torch.device("cpu")
        )
        model.load_state_dict(state)  # BaseOP: strict shape/dtype asserts + pop; raises on leftovers
    finally:
        _set_tp(info, 0, 1)


def test_rank0_state_dict_loads_the_model(ckpt_dir, loaded):
    """The strongest gate: every loader shard matches the shape the model builds under tp=2."""
    _load_model_state(loaded[0], ckpt_dir, 0)


def test_rank1_state_dict_loads_the_model(ckpt_dir, loaded):
    _load_model_state(loaded[1], ckpt_dir, 1)


# --------------------------------------------------------------------------------------
# NVFP4 expert banks: rank-sharded layout + pack
# --------------------------------------------------------------------------------------


def _nvfp4_pieces(batch: int):
    """Native modelopt pieces for one expert batch: gate/up [B, I, H/2], down [B, H, I/2]."""
    B = batch
    gate = torch.randint(0, 256, (B, I, H // 2), dtype=torch.uint8)
    up = torch.randint(0, 256, (B, I, H // 2), dtype=torch.uint8)
    down = torch.randint(0, 256, (B, H, I // 2), dtype=torch.uint8)
    gsc = torch.randint(1, 8, (B, I, H // 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
    usc = torch.randint(1, 8, (B, I, H // 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
    dsc = torch.randint(1, 8, (B, H, I // 16), dtype=torch.uint8).view(torch.float8_e4m3fn)
    return {
        "gate": gate, "up": up, "down": down,
        "gate_scale": gsc, "up_scale": usc, "down_scale": dsc,
        "gate_global": torch.full((B, 1), 0.5, dtype=torch.float16),
        "up_global": torch.full((B, 1), 0.25, dtype=torch.float16),
        "down_global": torch.full((B, 1), 0.125, dtype=torch.float16),
    }


def test_triton_bank_layout_is_local():
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    k = TritonNvfp4MoEKernel()
    for r in (0, 1):
        cfg = MoEConfig(num_experts=E, hidden=H, intermediate=I, top_k=2, tp_rank=r, tp_size=2, strategy="offload")
        lay = k.layout(cfg)
        il = I // 2
        assert tuple(lay["gate_up"].shape) == (2 * il, H // 2)
        assert tuple(lay["gate_up_scale"].shape) == (2 * il, H // 16)
        assert tuple(lay["down"].shape) == (H, il // 2)
        assert tuple(lay["down_scale"].shape) == (H, il // 16)
        assert k.unusable_reason(cfg) is None  # tp_ok now


def test_triton_pack_ranks_recat_to_full():
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    k = TritonNvfp4MoEKernel()
    full_cfg = MoEConfig(num_experts=E, hidden=H, intermediate=I, top_k=2, strategy="offload")
    full = k.layout(full_cfg)
    out_full = {
        role: torch.zeros(E, *spec.shape, dtype=spec.dtype)
        for role, spec in full.items()
    }
    pieces = _nvfp4_pieces(E)
    k.pack(pieces, full_cfg, out_full)

    outs = {}
    for r in (0, 1):
        cfg = MoEConfig(num_experts=E, hidden=H, intermediate=I, top_k=2, tp_rank=r, tp_size=2, strategy="offload")
        lay = k.layout(cfg)
        out = {role: torch.zeros(E, *spec.shape, dtype=spec.dtype) for role, spec in lay.items()}
        k.pack(pieces, cfg, out)
        outs[r] = out
    # rank banks hold [gate_loc | up_loc]; reconstruct the full [gate | up] block-wise
    for role in ("gate_up", "gate_up_scale", "gate_up_global"):
        il = I // 2
        assert torch.equal(torch.cat([outs[0][role][:, :il], outs[1][role][:, :il]], 1), out_full[role][:, :I]), role
        assert torch.equal(torch.cat([outs[0][role][:, il:], outs[1][role][:, il:]], 1), out_full[role][:, I:]), role
    for role in ("down", "down_scale"):
        assert torch.equal(torch.cat([outs[0][role], outs[1][role]], 2), out_full[role]), role
    assert torch.equal(outs[0]["down_global"], out_full["down_global"])


def test_marlin_bank_layout_is_local_and_tp_ok():
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.nvfp4 import MarlinNvfp4MoEKernel

    k = MarlinNvfp4MoEKernel()
    cfg = MoEConfig(num_experts=E, hidden=H, intermediate=I, top_k=2, tp_rank=1, tp_size=2, strategy="offload")
    lay = k.layout(cfg)
    il = I // 2
    assert tuple(lay["gate_up"].shape) == (H // 16, 4 * il)
    assert tuple(lay["gate_up_scale"].shape) == (H // 16, 2 * il)
    assert tuple(lay["down"].shape) == (il // 16, 2 * H)
    assert tuple(lay["down_scale"].shape) == (il // 16, H)
    # the TP rejection must be gone; on a CPU box the remaining vLLM gate may still speak
    reason = k.unusable_reason(cfg)
    assert reason is None or "TP > 1" not in reason, reason


# --------------------------------------------------------------------------------------
# Vision tower TP (Qwen3-VL ViT): qkv head-split + bias, col/row MLP shards
# --------------------------------------------------------------------------------------

VHID, VHEADS, VI, VOUT = 32, 4, 24, 32  # merged = VHID * spatial_merge^2 = 128


def _vision_cfg():
    return SimpleNamespace(
        hidden_size=VHID, depth=1, num_heads=VHEADS, intermediate_size=VI,
        patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=16, out_hidden_size=VOUT, in_channels=3,
        deepstack_visual_indexes=[],
    )


def _vision_raw() -> dict[str, torch.Tensor]:
    v = "model.visual"
    return {
        f"{v}.blocks.0.attn.qkv.weight": _bf16(3 * VHID, VHID),
        f"{v}.blocks.0.attn.qkv.bias": _bf16(3 * VHID),
        f"{v}.blocks.0.attn.proj.weight": _bf16(VHID, VHID),
        f"{v}.blocks.0.attn.proj.bias": _bf16(VHID),
        f"{v}.blocks.0.mlp.linear_fc1.weight": _bf16(VI, VHID),
        f"{v}.blocks.0.mlp.linear_fc1.bias": _bf16(VI),
        f"{v}.blocks.0.mlp.linear_fc2.weight": _bf16(VHID, VI),
        f"{v}.blocks.0.mlp.linear_fc2.bias": _bf16(VHID),
        f"{v}.blocks.0.norm1.weight": _bf16(VHID), f"{v}.blocks.0.norm1.bias": _bf16(VHID),
        f"{v}.blocks.0.norm2.weight": _bf16(VHID), f"{v}.blocks.0.norm2.bias": _bf16(VHID),
        f"{v}.merger.linear_fc1.weight": _bf16(VHID * 4, VHID * 4),
        f"{v}.merger.linear_fc1.bias": _bf16(VHID * 4),
        f"{v}.merger.linear_fc2.weight": _bf16(VOUT, VHID * 4),
        f"{v}.merger.linear_fc2.bias": _bf16(VOUT),
        f"{v}.merger.norm.weight": _bf16(VHID), f"{v}.merger.norm.bias": _bf16(VHID),
        f"{v}.patch_embed.proj.weight": _bf16(VHID, 3, 2, 16, 16),
        f"{v}.patch_embed.proj.bias": _bf16(VHID),
        f"{v}.pos_embed.weight": _bf16(16, VHID),
    }


@pytest.fixture(scope="module")
def vision_ckpt(tmp_path_factory):
    torch.manual_seed(11)
    folder = tmp_path_factory.mktemp("qwen4_exp_tp_vision")
    raw = _raw_checkpoint()
    raw.update(_vision_raw())
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    cfg = _text_cfg()
    cfg.vision_config = _vision_cfg()
    with open(folder / "config.json", "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(cfg), fh)
    return str(folder)


@pytest.fixture(scope="module")
def loaded_vision(vision_ckpt, tp_global):
    from freetoken.distributed import info
    from freetoken.models.qwen4_exp.weight import iter_weights

    out = {}
    for key, (rank, size) in {"full": (0, 1), 0: (0, 2), 1: (1, 2)}.items():
        _set_tp(info, rank, size)
        out[key] = {
            n: t.clone()
            for n, t in iter_weights(
                vision_ckpt, torch.device("cpu"),
                include_moe_experts=True, include_non_moe=True, include_vision=True,
            )
            if n.startswith("visual.")
        }
    return out


def test_vision_shards_recat_to_full(loaded_vision):
    full, r0, r1 = loaded_vision["full"], loaded_vision[0], loaded_vision[1]
    assert set(full) == set(r0) == set(r1)
    assert len(full) == len(_vision_raw())
    seg = VHID  # n_heads * head_dim with tiny geometry
    nl = VHEADS // 2
    hd = VHID // VHEADS
    for n, t in full.items():
        if n.endswith((".attn.qkv.weight", ".attn.qkv.bias")):
            assert r0[n].shape[0] == 3 * nl * hd
            for lo, hi, flo in ((0, nl * hd, 0), (nl * hd, 2 * nl * hd, seg), (2 * nl * hd, 3 * nl * hd, 2 * seg)):
                assert torch.equal(torch.cat([r0[n][lo:hi], r1[n][lo:hi]], 0), t[flo : flo + 2 * (hi - lo)])
        elif n.endswith((".attn.proj.weight", ".mlp.linear_fc2.weight", ".merger.linear_fc2.weight")):
            assert torch.equal(torch.cat([r0[n], r1[n]], 1), t)
        elif n.endswith((".mlp.linear_fc1.weight", ".mlp.linear_fc1.bias",
                         ".merger.linear_fc1.weight", ".merger.linear_fc1.bias")):
            assert torch.equal(torch.cat([r0[n], r1[n]], 0), t)
        else:  # norms, patch_embed, pos_embed: replicated byte-for-byte
            assert torch.equal(r0[n], t) and torch.equal(r1[n], t)


def test_vision_shards_match_module_shapes(loaded_vision, tp_global):
    from freetoken.distributed import info
    from freetoken.models.qwen3_vl.config import VisionConfig
    from freetoken.models.qwen3_vl.vision import VisionAttention, VisionMLP, VisionPatchMerger

    _set_tp(info, 0, 2)
    try:
        vc = VisionConfig(
            hidden_size=VHID, depth=1, num_heads=VHEADS, intermediate_size=VI, patch_size=16,
            temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=16,
            out_hidden_size=VOUT, in_channels=3,
        )
        expected: dict[str, tuple[int, ...]] = {}
        for mod, key_of in (
            (VisionAttention(vc), lambda k: f"visual.blocks.0.attn.{k}"),
            (VisionMLP(vc), lambda k: f"visual.blocks.0.mlp.{k}"),
            (VisionPatchMerger(vc), lambda k: f"visual.merger.{k}"),
        ):
            for k, tensor in mod.state_dict().items():
                expected[key_of(k)] = tuple(tensor.shape)
        r0 = loaded_vision[0]
        for name, shape in expected.items():
            assert name in r0, f"loader never emitted {name}"
            assert tuple(r0[name].shape) == shape, f"{name}: loader {tuple(r0[name].shape)} != module {shape}"
    finally:
        _set_tp(info, 0, 1)
