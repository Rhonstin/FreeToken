"""NVFP4 KV against independent torch rounding and dense attention references."""

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache.mha_pool import MHAKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _pool(dim=128, heads=2, slots=96, layer_ids=None):
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    return MHAKVCache(heads, 4, dim, slots // 4, 4, torch.bfloat16,
                      torch.device("cuda"), layer_ids=layer_ids, kv_quant="nvfp4")


def _reference(x):
    shape = x.shape
    x = x.float().reshape(*shape[:-1], -1, 16)
    row = x.abs().flatten(-2).amax(-1).clamp_min(1e-10) / 2688.0
    block = (x.abs().amax(-1) / (6 * row[..., None])).clamp_max(448).to(torch.float8_e4m3fn)
    denom = block.float() * row[..., None]
    normalized = torch.where(denom[..., None] > 0, x / denom[..., None], 0)
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    distance = (normalized.abs()[..., None] - grid).abs()
    nearest = distance == distance.amin(-1, keepdim=True)
    codes = torch.arange(8, device=x.device).expand_as(distance)
    # Prefer even codes on ties, independently of the kernel's threshold encoding.
    rank = torch.where(nearest, codes % 2 * 8 + codes, 32)
    code = rank.argmin(-1) | ((normalized < 0).long() * 8)
    code = code.reshape(shape)
    packed = (code[..., ::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    return packed, block.view(torch.uint8), row


def _decode(pool, which, layer=1):
    codes = getattr(pool, f"{which}_cache")(layer).flatten(0, 1)
    block = getattr(pool, f"{which}_block_scale")(layer).view(torch.float8_e4m3fn).float()
    row = getattr(pool, f"{which}_scale")(layer)
    code = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                         0, -.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
    return grid[code] * block.repeat_interleave(16, -1) * row[..., None]


def _decode_latent(pool, layer=0):
    codes = pool.latent_rows(layer)
    code = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    grid = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                         0, -.5, -1, -1.5, -2, -3, -4, -6], device=codes.device)
    block = pool.latent_block_scale(layer).view(torch.float8_e4m3fn).float()
    return grid[code] * block.repeat_interleave(16, -1) * pool.latent_scale(layer)[:, None]


@pytest.mark.parametrize("dim", [16, 48, 64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scatter_matches_independent_reference(dim, dtype):
    torch.manual_seed(41)
    pool = _pool(dim)
    # K is a projection slice; V is independently contiguous.
    k = torch.randn(5, 4 * dim, device="cuda", dtype=dtype)[:, :2 * dim]
    v = torch.randn(5, 2 * dim, device="cuda", dtype=dtype)
    k[0].zero_()
    k[1, :16] *= 100
    v[2] *= 1e-5
    loc = torch.tensor([7, 1, 95, 31, 12], device="cuda", dtype=torch.int64)
    pool.store_kv(k, v, loc, 1)
    for which, source in (("k", k), ("v", v)):
        packed, block, row = _reference(source.reshape(5, 2, dim))
        torch.testing.assert_close(getattr(pool, f"{which}_cache")(1).flatten(0, 1)[loc], packed)
        torch.testing.assert_close(getattr(pool, f"{which}_block_scale")(1)[loc], block)
        torch.testing.assert_close(getattr(pool, f"{which}_scale")(1)[loc], row)
        assert torch.isfinite(_decode(pool, which)).all()
        assert torch.count_nonzero(_decode(pool, which)[0]) == 0


def test_e2m1_grid_and_round_to_even_boundaries():
    positive = torch.tensor([0, .25, .5, .75, 1, 1.25, 1.5, 1.75,
                             2, 2.5, 3, 3.5, 4, 5, 6], device="cuda")
    values = torch.cat((positive, -positive))
    values = torch.cat((values, values.nextafter(torch.full_like(values, float("inf"))),
                        values.nextafter(torch.full_like(values, -float("inf")))))
    pool = _pool(dim=32, slots=192)
    rows = torch.zeros(values.numel(), 2, 32, device="cuda")
    rows[:, :, 0] = values[:, None]
    rows[:, :, 15] = 6
    rows[:, :, 31] = 2688  # Forces row_scale=1 and the first block_scale=1.
    loc = torch.arange(values.numel(), device="cuda", dtype=torch.int32)
    pool.store_kv(rows.flatten(1), rows.flatten(1), loc, 1)
    packed, block, row = _reference(rows)
    torch.testing.assert_close(pool.k_cache(1).flatten(0, 1)[loc], packed)
    torch.testing.assert_close(pool.k_block_scale(1)[loc], block)
    torch.testing.assert_close(pool.k_scale(1)[loc], row)


def test_pool_budget_rebuild_and_layer_mapping():
    from freetoken.kvcache.base import spec_kv_bytes_per_token
    from freetoken.models.config import KVCacheGroupSpec

    pool = _pool(layer_ids=(1, 3))
    spec = KVCacheGroupSpec(name="full", layer_ids=(1, 3), num_kv_heads=2, head_dim=128, sliding_window=None)
    cfg = SimpleNamespace(kv_quant="nvfp4", dtype=torch.bfloat16, tp_info=SimpleNamespace(size=1))
    assert spec_kv_bytes_per_token(spec, cfg) == pool.unit_bytes()[0] == 2 * 2 * 2 * 76
    with pytest.raises(KeyError):
        pool.k_block_scale(0)
    for pages in (32, 8):
        pool.rebuild(pages)
        assert pool.k_cache(3).shape == (pages, 4, 2, 64)
        assert pool.k_block_scale(3).shape == (pages * 4, 2, 8)
        assert pool.unit_bytes()[0] == 608
        assert torch.count_nonzero(_decode(pool, "k", 3)) == 0


def test_store_cuda_graph_replay_changes_slots():
    pool = _pool()
    k = torch.randn(2, 256, device="cuda", dtype=torch.bfloat16)
    loc = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    pool.store_kv(k, k, loc, 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool.store_kv(k, k, loc, 1)
    k.mul_(2)
    loc.copy_(torch.tensor([4, 9], device="cuda", dtype=torch.int32))
    graph.replay()
    expected, _, _ = _reference(k.view(2, 2, 128))
    torch.testing.assert_close(pool.k_cache(1).flatten(0, 1)[loc], expected)


# NOTE: the attention/latent readers for nvfp4 (test_attention_reads_packed_cache,
# test_sparse_mla_nvfp4..., test_latent_scatter..., hybrid-SWA packing and the backend
# decode-graph test) land with task 5y2.15; this file covers the codec, the scatter
# writer and the memory contract only.


def test_nvfp4_memory_contract_d128_row_is_76_bytes():
    """D=128 row: 64 packed code bytes + 8 E4M3 block scales + 4 B FP32 row scale = 76;
    a K/V pair is 152 B per token, and the planner prices exactly what the pool allocates."""
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    heads, layers, pages, page_size = 2, 4, 8, 32
    pool = MHAKVCache(
        num_kv_heads=heads, num_layers=layers, head_dim=128, num_pages=pages,
        page_size=page_size, dtype=torch.bfloat16, device=torch.device("cuda"),
        kv_quant="nvfp4",
    )
    tokens = pages * page_size
    kv_bytes, _ = pool.unit_bytes()
    assert kv_bytes == 2 * layers * heads * 76, kv_bytes
    assert pool.k_cache(0).element_size() == 1
    assert pool.k_cache(0).shape[-1] == 64
    assert pool.k_block_scale(0).shape[-1] == 8
    assert pool.k_scale(0).dtype is torch.float32
    assert pool.k_block_scale(0).dtype is torch.uint8


def test_nvfp4_rejects_head_dim_not_divisible_by_16():
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    with pytest.raises(ValueError, match="divisible by 16"):
        MHAKVCache(
            num_kv_heads=2, num_layers=4, head_dim=40, num_pages=2, page_size=16,
            dtype=torch.bfloat16, device=torch.device("cuda"), kv_quant="nvfp4",
        )
