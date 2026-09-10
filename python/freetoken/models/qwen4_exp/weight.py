"""Qwen3.8-Flash-Next checkpoint reader (the NVFP4 and the official block-fp8 releases).

Three separate paths, because the checkpoint's three weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`nvfp4_expert_spec` -- how the routed NVFP4 experts are named, for the offload cache's expert reader.
* :func:`iter_mtp_expert_pieces` -- the MTP head's stacked block-fp8 routed experts, as per-expert pieces for the offload banks (``iter_weights`` skips them).

``iter_weights`` drops ``mtp.*`` unless asked: the MTP speculative head (a full decoder
layer plus a top-level hyper-connection mixer, keys already ``mtp.``-namespaced) is only
loaded on request (``include_mtp=True``), gated by the draft module the model builds. Its
routed experts are stacked per-MTP-layer dense-bf16 tensors, always served from the
expert banks instead of the state dict. ``model.visual.*`` stays dropped (served text-only).
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import ShardReader, drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")

# The MTP head's stacked dense-bf16 routed experts: one fused tensor per role per MTP
# layer (no per-block scales -- unlike the NVFP4 target experts, these are plain bf16).
_MTP_EXPERT_RE = re.compile(
    r"^mtp\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<name>gate_up_proj|down_proj)$"
)
_MTP_EXPERT_BATCH = 32  # experts per piece; bounds the transient slice footprint
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"
_PLE_FILE_BYTES = 4 << 30  # ple-table-*.safetensors written by ftw_side_files

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# Fused projections: concat the checkpoint parts along dim 0 in this exact order. A nonzero pad
# rounds the merged row count up; the model splits the result back with the same sizes.
_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    # q carries the output gate, so its half is twice the attention width: [2*qo | kv | kv].
    ".self_attn.qkv_proj.weight": ((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ), 0),
    ".linear_attn.in_proj.weight": ((
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ), 0),
    ".mlp.shared_expert.gate_up_proj.weight": ((
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ), 0),
    # HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM
    # pads the merged output to a multiple of 16 rows for cuBLAS (hyperconnection.py pad_size).
    # The top-level hyper_connection_mixer has no injection and so never fuses.
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
    ), 16),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
    ), 16),
}


def _rename(raw_name: str, *, include_mtp: bool = False) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith("mtp."):
        if not include_mtp:
            return None
        if _MTP_EXPERT_RE.match(raw_name):
            return None  # MTP routed experts: iter_mtp_expert_pieces
        return raw_name  # already namespaced below the model root
    if raw_name.startswith(("model.visual.", "visual.")):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
            if pad:
                rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
            return key, torch.cat(rows, dim=0)
    return None


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
    include_mtp: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Nothing here is quantized: every release's skip
    list (modelopt ``ignore``, fp8 ``modules_to_not_convert``) covers everything except those experts,
    so attention, GDN, HC, PLE, the shared expert and lm_head are all plain bf16 (the n-gram hash
    constants stay int64). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from the offload cache's expert reader.

    ``include_mtp`` adds the MTP head's dense tensors under their ``mtp.`` keys (fusions apply
    inside the namespace the same way); its routed experts never flow through here. Default off:
    the model only asks for the head once it builds the draft module, and load_state_dict is
    strict about unexpected keys.
    """
    if get_tp_info().size > 1:
        raise NotImplementedError("qwen4_exp weight loading supports TP=1 only")
    if not include_non_moe:
        return

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name, include_mtp=include_mtp)
                if name is None:
                    continue
                tensor = f.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused
                    continue
                yield name, tensor

    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor FP8 scale."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` float8_e4m3fn view of the bank."""
        return self.bank.tensor


_PLE_ST_DTYPE = "F8_E4M3"


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def ftw_side_files(model_path: str, out_dir: str) -> list[str]:
    """Write the PLE n-gram table tensors, and only those, into ``ple-table-*.safetensors`` next to an FTW checkpoint.

    The table is served from safetensors files in the checkpoint dir (see load_ple_table), not from FTW entries."""
    from safetensors.torch import save_file

    folder = download_hf_weight(model_path)
    written: list[str] = []
    batch: dict[str, torch.Tensor] = {}
    size = 0

    def flush():
        nonlocal batch, size
        if batch:
            name = f"ple-table-{len(written):05d}.safetensors"
            save_file(batch, os.path.join(out_dir, name))
            written.append(name)
            batch, size = {}, 0

    for path in _ple_table_files(folder):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if _PLE_TABLE_INFIX not in key:
                    continue
                t = f.get_tensor(key)
                batch[key] = t
                size += t.numel() * t.element_size()
                if size >= _PLE_FILE_BYTES:
                    flush()
    flush()
    return written


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.
    """
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")

    bank = HostBank((expected * rows, cols), torch.float8_e4m3fn)
    shard_bytes = rows * cols
    bar = byte_bar(expected * shard_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            read_range_into(buf, path, file_offset=offset, nbytes=nbytes,
                            dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


# ======================================================================================
# MTP head: stacked block-fp8 experts + the quant mapping
# ======================================================================================


def _mtp_expert_files(model_path: str) -> dict[str, str]:
    """MTP stacked-expert checkpoint key -> shard path (index when there is one, headers otherwise)."""
    folder = download_hf_weight(model_path)
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        return {k: os.path.join(folder, v) for k, v in weight_map.items() if _MTP_EXPERT_RE.match(k)}
    files: dict[str, str] = {}
    for file in iter_weight_files(folder):
        header, _base = _safetensors_header(file)
        files.update({k: file for k in header if k != "__metadata__" and _MTP_EXPERT_RE.match(k)})
    return files


def iter_mtp_expert_pieces(model_path: str, model_config, *, batch: int = _MTP_EXPERT_BATCH):
    """The MTP head's stacked dense-bf16 routed experts as per-expert pieces for ``build_expert_banks``.

    The head stores one fused tensor per role per MTP layer (``mtp.layers.N.mlp.experts.
    gate_up_proj`` [E, 2I, H] bf16 and the ``down_proj`` [E, H, I] bf16 -- checkpoint
    ground truth, no ``_scale_inv`` companions), so pieces slice the gate/up halves out
    of the fused rows. Roles carry the target's bf16 bank names ({gate, up, down}),
    packed through the draft layers' own (unquantized) method; layer ids continue past
    the target's MoE layers (``num_moe_layers + mtp_layer``) so the attach path can key
    them. A non-floating (quantized) stacked expert fails loudly: the bf16 bank pack
    cannot serve it.
    """
    keys = _mtp_expert_files(model_path)
    if not keys:
        return
    E = model_config.num_experts
    I = model_config.moe_intermediate_size
    H = model_config.hidden_size
    layers = sorted({int(_MTP_EXPERT_RE.match(k).group("layer")) for k in keys})
    reader = ShardReader(model_path, torch.device("cpu"))
    try:
        for li in layers:
            base = f"mtp.layers.{li}.mlp.experts"
            gate_up = reader.get_tensor(f"{base}.gate_up_proj")
            down = reader.get_tensor(f"{base}.down_proj")
            for name, tensor, shape in (
                ("gate_up_proj", gate_up, (E, 2 * I, H)),
                ("down_proj", down, (E, H, I)),
            ):
                if tuple(tensor.shape) != shape:
                    raise ValueError(f"{base}.{name} is {tuple(tensor.shape)}, expected {shape}")
            if not (gate_up.is_floating_point() and down.is_floating_point()):
                raise ValueError(
                    f"{base}: stacked MTP experts must be dense floating point, "
                    f"got {gate_up.dtype} / {down.dtype}")
            for e0 in range(0, E, batch):
                e1 = min(e0 + batch, E)
                yield model_config.num_moe_layers + li, e0, e1, {
                    "gate": gate_up[e0:e1, :I],
                    "up": gate_up[e0:e1, I:],
                    "down": down[e0:e1],
                }
    finally:
        reader.close()


def mtp_expert_method(model_config):
    """The expert quant method the MTP head's routed experts pack through.

    Built through the #418 layers (``fp8_block_scheme`` -> ``method_class`` -> backend) with
    the target's expert geometry, so its bank layout is exactly the target offload layers':
    one bank schema, one cache.

    NOTE: the shipping RadixArk NVFP4 checkpoint carries dense-BF16 stacked MTP experts,
    not block-fp8 -- for that checkpoint the attach path packs through the draft layers'
    own (unquantized) method instead (see ``build_mtp_draft_banks``). This constructor
    stays for block-fp8 releases whose stacked experts do carry ``_scale_inv`` companions.
    """
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.quant_backend import get_quant_backend
    from freetoken.layers.quantization.registry import LayerKind, method_class
    from freetoken.layers.quantization.scheme import QuantKind, fp8_block_scheme

    cfg = MoEConfig(
        num_experts=model_config.num_experts,
        hidden=model_config.hidden_size,
        intermediate=model_config.moe_intermediate_size,
        top_k=model_config.num_experts_per_tok,
        scheme=fp8_block_scheme("float"),
        activation=model_config.hidden_act,
        strategy="offload",
    )
    cls = method_class(QuantKind.FP8_BLOCK, LayerKind.MOE)
    return cls(cfg, get_quant_backend().select(LayerKind.MOE, QuantKind.FP8_BLOCK))


def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


__all__ = [
    "iter_mtp_expert_pieces",
    "mtp_expert_method",
    "nvfp4_expert_spec",
    "PleTable",
    "iter_weights",
    "load_ple_table",
]
