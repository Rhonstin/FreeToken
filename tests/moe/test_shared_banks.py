"""Shared expert-bank arena: offset-stable layout and cross-handle sharing (CPU only)."""

from __future__ import annotations

import torch

from freetoken.moe.host_banks import (
    SharedArena,
    align_up,
    alloc_banks,
    alloc_layer_banks,
    arena_tensor,
)


def test_arena_layer_banks_share_one_file_with_stable_offsets(tmp_path):
    path = str(tmp_path / "banks.shm")
    specs = {"w": ((4, 8), torch.float32), "s": ((4, 4), torch.uint8)}

    a1 = SharedArena(path)
    hb1 = alloc_layer_banks(specs, num_layers=3, arena=a1)
    for role, banks in hb1.items():
        for layer, bank in enumerate(banks):
            if bank.tensor.dtype.is_floating_point:
                bank.tensor.fill_(float(layer + 1))
            else:
                bank.tensor.view(torch.uint8).fill_(layer + 7)
    a1.close()

    # a second handle over the same file must land on the same offsets and see the same bytes
    a2 = SharedArena(path)
    hb2 = alloc_layer_banks(specs, num_layers=3, arena=a2)
    for role in specs:
        for b1, b2 in zip(hb1[role], hb2[role]):
            assert torch.equal(b1.tensor, b2.tensor)
    a2.close()


def test_arena_never_shrinks_on_ensure_size(tmp_path):
    path = str(tmp_path / "banks.shm")
    a = SharedArena(path)
    a.ensure_size(1 << 20)
    a.ensure_size(1 << 16)  # smaller must not truncate
    assert a._size == (1 << 20)
    a.close()


def test_arena_tensor_and_alloc_banks_share_arena(tmp_path):
    path = str(tmp_path / "banks.shm")
    a = SharedArena(path)
    sc = arena_tensor(a, (6,), torch.float32)
    sc.fill_(3.5)
    banks = alloc_banks({"b": ((2, 6), torch.float32)}, arena=a)
    banks["b"].tensor.fill_(1.0)
    assert torch.equal(sc, torch.full((6,), 3.5))
    assert torch.equal(banks["b"].tensor, torch.ones(2, 6))
    a.close()


def test_align_up_to_page():
    assert align_up(1) == 4096
    assert align_up(4096) == 4096
    assert align_up(4097) == 8192
