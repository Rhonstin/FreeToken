"""P0 — global-paged SWA pool (Option A) allocator semantics.

CPU-only. Covers the pieces that must match sglang's SWATokenToKVPoolAllocator:
the dense full->swa mapping with the slot-0 sentinel, alloc_swa + mapping-based
translate, free_swa idempotence over the sentinel (no double-free), exhaustion,
and the rebuild reset (mapping + free-list re-sized atomically with the buffer).
"""
from __future__ import annotations

import pytest
import torch


def _patch_tp(monkeypatch) -> None:
    from freetoken.distributed.info import DistributedInfo

    monkeypatch.setattr(
        "freetoken.kvcache.hybrid_swa_pool.get_tp_info",
        lambda: DistributedInfo(rank=0, size=1),
    )


def _specs():
    from freetoken.models.config import KVCacheGroupSpec

    return (
        KVCacheGroupSpec(name="full", layer_ids=(1,), num_kv_heads=1, head_dim=8, sliding_window=None),
        KVCacheGroupSpec(name="swa", layer_ids=(0,), num_kv_heads=1, head_dim=8, sliding_window=4),
    )


def _paged_pool(num_full=16, num_swa=8, kv_quant="none", device=None):
    from freetoken.kvcache.hybrid_swa_pool import HybridSWAKVCache

    return HybridSWAKVCache(
        groups=_specs(),
        num_layers=2,
        num_full_pages=num_full,  # page_size=1 -> full_num_tokens == num_full
        page_size=1,
        dtype=torch.bfloat16,
        device=device or torch.device("cpu"),
        num_swa_tokens=num_swa,
        kv_quant=kv_quant,
    )


def _allocated_mask(pool) -> torch.Tensor:
    """full slots whose mapping is a live swa slot (> 0)."""
    return pool.full_to_swa_index_mapping[: pool.full_num_tokens] > 0


def test_construction_mapping_and_freelist(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)

    assert pool.swa_paged
    # mapping: full_num_tokens + page_size zeros, trailing -1 sentinel for last_loc == -1.
    assert pool.full_to_swa_index_mapping.numel() == 16 + 1 + 1
    assert pool.full_to_swa_index_mapping.dtype == torch.int64
    assert int(pool.full_to_swa_index_mapping[-1]) == -1
    assert torch.all(pool.full_to_swa_index_mapping[:-1] == 0)
    # slot 0 reserved as the "no live swa slot" sentinel -> 7 allocatable of 8.
    assert pool.swa_available_size() == 7


def test_alloc_swa_writes_mapping_and_translate_reads_it(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)

    full = torch.tensor([0, 3, 5], dtype=torch.int32)
    pool.alloc_swa(full)
    assert pool.swa_available_size() == 7 - 3

    swa = pool.translate_loc_from_full_to_swa(full)
    assert swa.dtype == torch.int32
    # every mapped slot is a distinct, in-range, non-sentinel swa slot.
    assert torch.all(swa >= 1) and torch.all(swa < 8)
    assert len(set(swa.tolist())) == 3
    # unallocated full slots translate to the 0 sentinel.
    assert int(pool.translate_loc_from_full_to_swa(torch.tensor([7], dtype=torch.int32))[0]) == 0

    # slot conservation: free + live-mapped == total allocatable, always.
    assert pool.swa_available_size() + int(_allocated_mask(pool).sum()) == 7


def test_free_swa_is_idempotent_over_sentinel(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)

    full = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    pool.alloc_swa(full)
    assert pool.swa_available_size() == 3

    pool.free_swa(torch.tensor([0, 1], dtype=torch.int32))
    assert pool.swa_available_size() == 5  # 2 returned
    assert int(pool.full_to_swa_index_mapping[0]) == 0
    assert int(pool.full_to_swa_index_mapping[1]) == 0

    # freeing the same (now-sentinel) slots again must be a no-op: filter > 0, no double-free.
    pool.free_swa(torch.tensor([0, 1], dtype=torch.int32))
    assert pool.swa_available_size() == 5
    assert pool.swa_available_size() + int(_allocated_mask(pool).sum()) == 7


def test_free_then_realloc_reuses_slots_no_leak(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)

    pool.alloc_swa(torch.arange(7, dtype=torch.int32))  # exhaust all 7 allocatable
    assert pool.swa_available_size() == 0
    pool.free_swa(torch.arange(7, dtype=torch.int32))
    assert pool.swa_available_size() == 7
    assert torch.all(pool.full_to_swa_index_mapping[:-1] == 0)
    # can fully re-allocate after freeing -> no leak.
    pool.alloc_swa(torch.tensor([10, 11, 12, 13, 14, 15, 9], dtype=torch.int32))
    assert pool.swa_available_size() == 0


def test_alloc_swa_exhaustion_raises(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)
    with pytest.raises(RuntimeError, match="SWA pool exhausted"):
        pool.alloc_swa(torch.arange(8, dtype=torch.int32))  # only 7 allocatable


def test_rebuild_resets_mapping_and_freelist(monkeypatch):
    _patch_tp(monkeypatch)
    pool = _paged_pool(num_full=16, num_swa=8)
    pool.alloc_swa(torch.tensor([0, 1, 2], dtype=torch.int32))
    assert pool.swa_available_size() == 4

    pool.rebuild(num_full_pages=32, num_swa_tokens=12)

    assert pool.full_num_tokens == 32
    assert pool.swa_num_tokens == 12
    assert pool.full_to_swa_index_mapping.numel() == 32 + 1 + 1
    assert int(pool.full_to_swa_index_mapping[-1]) == -1
    assert torch.all(pool.full_to_swa_index_mapping[:-1] == 0)  # all stale mappings cleared
    assert pool.swa_available_size() == 11  # fresh free-list at new size, no leak
    # allocator still works at the new geometry.
    pool.alloc_swa(torch.tensor([30, 31], dtype=torch.int32))
    assert pool.swa_available_size() == 9


def test_fp8_pool_separates_compute_and_store_dtype(monkeypatch):
    """Both groups shrink to codes, and the scale views follow the layer mapping."""
    _patch_tp(monkeypatch)
    from freetoken.kernel.triton.kv_quant import kv_codes_dtype

    quantized = _paged_pool(kv_quant="fp8")
    plain = _paged_pool()
    assert quantized.dtype is torch.bfloat16 and plain.dtype is torch.bfloat16
    assert quantized.store_dtype == kv_codes_dtype()
    assert plain.store_dtype is torch.bfloat16
    # layer 0 -> swa group, layer 1 -> full group; the buffers really shrank.
    assert quantized.k_cache(1).element_size() == 1
    assert quantized.k_cache(0).element_size() == 1
    assert plain.k_cache(1).element_size() == 2
    # Scales have the same physical ownership as the payload they describe.
    assert quantized.k_scale(1).shape == (16, 1)  # full tokens
    assert quantized.k_scale(0).shape == (8, 1)  # swa tokens
    assert quantized.k_scale(1).dtype is torch.float32
    assert plain.k_scale(1) is None and plain.v_scale(0) is None


def test_fp8_unit_bytes_prices_the_scale_sidecar_in_both_groups(monkeypatch):
    """fp8 = 1 code byte + 4 scale bytes per (token, head) vs bf16's 2 bytes.

    At this fixture's head_dim 8 the sidecar is large (4/8 of the codes), which is
    exactly the accounting contract: both groups must price it, and the sum must match
    the buffers the planner sized."""
    _patch_tp(monkeypatch)
    assert _paged_pool().unit_bytes() == (32, 32)
    assert _paged_pool(kv_quant="fp8").unit_bytes() == (24, 24)


def test_fp8_rebuild_resizes_codes_and_scales_together(monkeypatch):
    _patch_tp(monkeypatch)
    from freetoken.kernel.triton.kv_quant import kv_codes_dtype

    pool = _paged_pool(kv_quant="fp8")
    pool.rebuild(num_full_pages=32, num_swa_tokens=12)
    assert pool.store_dtype == kv_codes_dtype()
    assert pool.full_num_tokens == 32 and pool.swa_num_tokens == 12
    assert pool.k_cache(1).shape[0] == 32 and pool.k_scale(1).shape == (32, 1)
    assert pool.k_cache(0).shape[0] == 12 and pool.k_scale(0).shape == (12, 1)
    # Fresh buffers are zero-filled, so an unwritten code decodes to 0.0, not NaN.
    assert (pool.k_cache(1) == 0).all() and (pool.k_cache(0) == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 scatter needs CUDA")
def test_fp8_store_kv_translates_slots_and_scatters_codes_and_scales(monkeypatch):
    """The SWA group's write goes through translate_loc_from_full_to_swa, so codes and
    scales must land on the TRANSLATED physical slot, not the logical one; an
    unmapped (sentinel) slot stays exactly zero."""
    from freetoken.kernel.triton.kv_quant import codes_to_f32

    _patch_tp(monkeypatch)
    dev = torch.device("cuda")
    pool = _paged_pool(kv_quant="fp8", device=dev)
    k = torch.randn(3, 8, device=dev, dtype=torch.bfloat16) * 2.0
    v = torch.randn_like(k)
    full_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=dev)
    pool.alloc_swa(full_loc)

    pool.store_kv(k, v, full_loc, layer_id=0)  # swa group
    torch.cuda.synchronize()
    swa_loc = pool.translate_loc_from_full_to_swa(full_loc).long()
    assert pool.k_scale(0).shape == (8, 1)
    scales_view = pool.k_scale(0)[swa_loc]
    # The pool hands out the RAW buffer (slots, inner, heads, dim); the attention
    # backend flattens it. This fixture has inner=1, heads=1, so index both away.
    codes = pool.k_cache(0)[swa_loc][:, 0, 0, :]  # (tokens, head_dim)
    scales = scales_view[:, 0]  # (tokens,)
    deq = codes_to_f32(codes) * scales.unsqueeze(-1)
    ref_scale = k.to(torch.float32).abs().amax(dim=-1) / 448.0
    torch.testing.assert_close(scales, ref_scale, rtol=1e-6, atol=0)
    assert torch.all(
        (deq - k.to(torch.float32)).abs()
        <= 0.08 * k.to(torch.float32).abs().amax(dim=-1, keepdim=True)
    )
    # The sentinel slot and every unmapped slot stayed zero (no stale codes).
    untouched = torch.ones(8, dtype=torch.bool, device=dev)
    untouched[swa_loc] = False
    assert (pool.k_cache(0)[untouched] == 0).all()
    assert (pool.k_scale(0)[untouched] == 0).all()

    # The full group stores by slot identity: same rows, same codes.
    pool.store_kv(k, v, full_loc, layer_id=1)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        pool.k_scale(1)[full_loc.long()], scales_view, rtol=0, atol=0
    )
    assert torch.equal(pool.k_cache(1)[full_loc.long()][:, 0, 0, :], codes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 scatter needs CUDA")
def test_fp8_reused_slot_gets_fresh_codes_and_scales(monkeypatch):
    """A slot handed out again must read back the NEW row: overwriting codes without
    overwriting the scale sidecar would silently serve the previous request's scale."""
    from freetoken.kernel.triton.kv_quant import codes_to_f32

    _patch_tp(monkeypatch)
    dev = torch.device("cuda")
    pool = _paged_pool(kv_quant="fp8", device=dev)
    loc = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    pool.alloc_swa(loc)
    first = torch.full((2, 8), 5.0, device=dev, dtype=torch.bfloat16)
    pool.store_kv(first, first, loc, layer_id=0)
    torch.cuda.synchronize()
    old_scale = pool.k_scale(0).clone()

    pool.free_swa(loc)
    pool.alloc_swa(loc)
    swa = pool.translate_loc_from_full_to_swa(loc).long()
    second = torch.full((2, 8), 0.25, device=dev, dtype=torch.bfloat16)
    pool.store_kv(second, second, loc, layer_id=0)
    torch.cuda.synchronize()

    scales = pool.k_scale(0)[swa][:, 0]
    ref = second.to(torch.float32).abs().amax(dim=-1) / 448.0
    torch.testing.assert_close(scales, ref, rtol=1e-6, atol=0)
    deq = codes_to_f32(pool.k_cache(0)[swa][:, 0, 0, :]) * scales.unsqueeze(-1)
    assert torch.all(
        (deq - second.to(torch.float32)).abs()
        <= 0.08 * second.to(torch.float32).abs().amax(dim=-1, keepdim=True)
    )
    # The write went through the byte view: the pool is still a real quantized pool.
    assert pool.store_dtype != torch.bfloat16 and old_scale.numel() > 0
