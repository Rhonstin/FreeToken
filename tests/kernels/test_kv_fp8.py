"""FP8 (e4m3) KV quantization: the store kernel against an independent oracle.

Expectations come from a brute-force nearest-code search over an e4m3 table decoded
from first principles (sign / exponent / mantissa), NOT from the kernels' own
rounding helpers -- so a drift in ``round_e4m3`` or in the new ``e4m3_f32_to_u8``
encoder fails here instead of being blessed by itself.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel.triton.kv_quant import (
    KV_SCALE_DTYPE,
    alloc_codes,
    codes_to_f32,
    kv_codes_dtype,
    quantize_kv_to_cache,
)

DEV = torch.device("cuda")
FP8_MAX = 448.0
CODE_448 = 0x7E


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


@pytest.fixture(autouse=True)
def _tp():
    _init_tp()


def _e4m3_table() -> dict[int, float]:
    """Every finite e4m3fn value, decoded by hand from its bit fields.

    code = S EEEE MMM, exponent bias 7: normal (E>0) -> (-1)^S * 2^(E-7) * (1+M/8);
    subnormal (E==0) -> (-1)^S * 2^-6 * (M/8). E==15 & M==7 (0x7F/0xFF) is the NaN
    pattern and is dropped, which caps the format at +-448.
    """
    values: dict[int, float] = {}
    for code in range(256):
        sign = -1.0 if (code >> 7) & 1 else 1.0
        exp, mant = (code >> 3) & 0x0F, code & 0x07
        if exp == 15 and mant == 7:
            continue
        values[code] = sign * (
            (2.0**-6) * (mant / 8.0) if exp == 0 else (2.0 ** (exp - 7)) * (1.0 + mant / 8.0)
        )
    return values


E4M3_VALUES = _e4m3_table()
CODES = torch.tensor(sorted(E4M3_VALUES), dtype=torch.int32)
GRID = torch.tensor([E4M3_VALUES[c] for c in CODES.tolist()], dtype=torch.float64)
NEGATIVE_ZERO = 0x80


def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    """A code buffer as raw bytes (the fp8 view on sm_89+, uint8 below it)."""
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _canonical_zero(codes: torch.Tensor) -> torch.Tensor:
    """Fold -0.0 (0x80) onto +0.0 (0x00).

    The two store paths legitimately disagree on zero's sign bit: sm_89+ converts with
    ``x.to(fp8e4nv)`` (keeps it), the emulated path rounds first and
    ``round_e4m3(-0.0)`` is documented to return +0.0. They decode to the same number,
    so a test that compares bytes must not care which one it got.
    """
    return torch.where(codes == NEGATIVE_ZERO, torch.zeros_like(codes), codes)


def _ref_codes(x: torch.Tensor) -> torch.Tensor:
    """Independent encoder: nearest grid value by brute force, ties resolved to the
    EVEN code (RNE). ``x`` is any shape; returns int32 codes."""
    grid = GRID.to(x.device)
    codes = CODES.to(x.device)
    dist = (x.to(torch.float64).reshape(-1, 1) - grid.unsqueeze(0)).abs()
    near = dist == dist.min(dim=-1, keepdim=True).values
    big = 1 << 30
    codes = codes.unsqueeze(0).expand_as(dist)
    any_code = torch.where(near, codes, torch.full_like(codes, big))
    even = near & (codes % 2 == 0)
    even_code = torch.where(even, codes, torch.full_like(codes, big))
    has_even = (even_code < big).any(dim=-1)
    return torch.where(has_even, even_code.min(dim=-1).values, any_code.min(dim=-1).values)


def _store(rows_k: torch.Tensor, rows_v: torch.Tensor):
    """Quantize ``[T, heads, dim]`` rows into a fresh code buffer, returning
    ``(k_codes, v_codes, k_scales, v_scales)``."""
    _init_tp()
    tokens, heads, dim = rows_k.shape
    k_cache = alloc_codes((tokens, heads, dim), DEV)
    v_cache = alloc_codes((tokens, heads, dim), DEV)
    k_scale = torch.zeros((tokens, heads), dtype=KV_SCALE_DTYPE, device=DEV)
    v_scale = torch.zeros((tokens, heads), dtype=KV_SCALE_DTYPE, device=DEV)
    quantize_kv_to_cache(
        k=rows_k.reshape(tokens, -1),
        v=rows_v.reshape(tokens, -1),
        out_loc=torch.arange(tokens, dtype=torch.int32, device=DEV),
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()
    return k_cache, v_cache, k_scale, v_scale


def test_scale_is_amax_over_e4m3_max():
    torch.manual_seed(0)
    k = (torch.randn(4, 2, 64, device=DEV, dtype=torch.bfloat16) * 3.0).to(torch.bfloat16)
    v = torch.randn_like(k)
    _, _, k_scale, v_scale = _store(k, v)
    # The kernel widens to fp32 before the amax/divide, so the reference must too:
    # a bf16 intermediate would round the expected scale and hide a precision bug.
    torch.testing.assert_close(
        k_scale, k.abs().to(torch.float32).amax(dim=-1) / FP8_MAX, rtol=1e-6, atol=0
    )
    torch.testing.assert_close(
        v_scale, v.abs().to(torch.float32).amax(dim=-1) / FP8_MAX, rtol=1e-6, atol=0
    )


def test_codes_match_the_reference_quantizer_and_reconstruction_is_close():
    torch.manual_seed(1)
    tokens, heads, dim = 8, 3, 128
    # Feed the rows as a qkv slice, the way the attention backends really do: the
    # row pitch is then wider than the row, which the store kernel must honour.
    qkv = torch.randn(tokens, heads * dim * 3, device=DEV, dtype=torch.bfloat16)
    qkv[:, heads * dim : 2 * heads * dim] *= 5.0  # K: large magnitude
    qkv[:, 2 * heads * dim :] *= 0.01  # V: subnormal end of the e4m3 grid
    _, k_rows, v_rows = qkv.split(heads * dim, dim=-1)
    k = k_rows.view(tokens, heads, dim)
    v = v_rows.view(tokens, heads, dim).clamp(-FP8_MAX, FP8_MAX)
    k_cache, v_cache, k_scale, v_scale = _store(k, v)

    for rows, cache, scale in ((k, k_cache, k_scale), (v, v_cache, v_scale)):
        f32 = rows.to(torch.float32)
        ref_scale = f32.abs().amax(dim=-1, keepdim=True) / FP8_MAX
        expected = _ref_codes((f32 / ref_scale).clamp(-FP8_MAX, FP8_MAX))
        got = _canonical_zero(_as_bytes(cache).reshape(-1).to(torch.int32))
        expected = _canonical_zero(expected)
        assert torch.equal(got, expected), (
            f"{int((got != expected).sum())} code mismatches of {got.numel()}"
        )
        deq = codes_to_f32(cache) * scale.unsqueeze(-1)
        err = (deq - f32).abs().max(dim=-1).values
        assert torch.all(err <= 0.08 * f32.abs().amax(dim=-1)), float(err.max())


def test_encoder_inverts_the_grid_through_the_scale_one_path():
    """Pack every e4m3 grid value into a row that also holds 448.0: the row scale is
    then exactly 1.0, so each stored byte IS the encoder's answer for that value."""
    dim, per_row = 256, 255
    pairs = sorted(E4M3_VALUES.items())
    tokens = -(-len(pairs) // per_row)
    rows = torch.zeros(tokens, 1, dim, dtype=torch.float32)
    expected = torch.zeros(tokens, dim, dtype=torch.uint8)
    for t in range(tokens):
        rows[t, 0, 0] = FP8_MAX  # the amax anchor
        expected[t, 0] = CODE_448
        for j in range(per_row):
            i = t * per_row + j
            if i >= len(pairs):
                break
            code, value = pairs[i]
            rows[t, 0, j + 1] = value
            expected[t, j + 1] = code

    k_cache, _, k_scale, _ = _store(
        rows.to(DEV, dtype=torch.bfloat16),
        torch.zeros(tokens, 1, dim, dtype=torch.bfloat16, device=DEV),
    )
    assert torch.equal(k_scale[:, 0], torch.ones_like(k_scale[:, 0]))
    got = _canonical_zero(_as_bytes(k_cache)[:, 0, :])
    want = _canonical_zero(expected.to(DEV))
    bad = got != want
    assert not bad.any().item(), (
        f"{int(bad.sum())} of {want.numel()} grid values round-tripped wrong; "
        f"first at {bad.nonzero()[0].tolist()}: expected "
        f"{want[bad][0].item():#x} got {got[bad][0].item():#x}"
    )


def test_zero_row_stays_finite_and_exact():
    k = torch.zeros(2, 2, 32, device=DEV, dtype=torch.bfloat16)
    k_cache, _, k_scale, _ = _store(k, k.clone())
    assert torch.isfinite(k_scale).all()
    assert (k_scale > 0).all(), "an all-zero row must still store a usable scale"
    assert (codes_to_f32(k_cache) == 0).all()


def test_codes_are_plain_bytes_and_the_kernel_decode_matches_torch():
    """The KV codec never puts an fp8 type in front of Triton.

    Codes live in a uint8 buffer on EVERY architecture and the kernel widens them with
    the software decoder, while the expectation below is torch's OWN e4m3 cast of those
    very bytes. That pins the one claim the design rests on: byte for byte, the
    software decode reads what a native fp8 unit would -- which is what lets the
    quantized cache behave identically on GPUs where the fp8 type is illegal.
    """
    import triton
    import triton.language as tl

    from freetoken.kernel.triton.e4m3_compat import kv_load_e4m3_tile_f32

    rows = torch.randn(5, 2, 64, device=DEV, dtype=torch.bfloat16) * 3.0
    k_cache, _, _, _ = _store(rows, rows.clone())
    assert kv_codes_dtype() is torch.uint8, "keep the fp8 type out of kernel signatures"
    assert k_cache.dtype is torch.uint8 and k_cache.element_size() == 1
    want = codes_to_f32(k_cache)  # torch reinterprets these bytes as e4m3 and casts

    @triton.jit
    def read_out(codes_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        got = kv_load_e4m3_tile_f32(codes_ptr + offs, offs < n)
        tl.store(out_ptr + offs, got, mask=offs < n)

    n = k_cache.numel()
    out = torch.zeros(n, dtype=torch.float32, device=DEV)
    read_out[(triton.cdiv(n, 256),)](k_cache, out, n, BLOCK=256)
    flat = want.reshape(-1)
    assert torch.equal(out, flat), (
        f"{int((out != flat).sum())} of {n} codes decode differently from torch's cast"
    )


def test_kv_codec_has_no_arch_or_dtype_branch():
    """The two rejected designs had one thing in common: they chose an arm.

    The compile-time fp8-native probe answers a question the allocator already
    answered -- and on one box answered wrongly -- while a test against the pointer's
    element type is NOT pruned by triton, so the dead arm still gets type-checked.
    That is how an int mask fill ended up in front of an fp8 pointer, twice. Both
    codecs are straight-line now; pin that, plus the identifiers of the two rejected
    designs, so neither creeps back in as a "fast path".
    """
    import inspect

    from freetoken.kernel.triton import kv_quant
    from freetoken.kernel.triton.e4m3_compat import kv_load_e4m3_tile_f32

    for obj in (kv_load_e4m3_tile_f32, kv_quant._kv_quant_scatter_kernel):
        src = inspect.getsource(getattr(obj, "fn", obj))
        for banned in ("e4m3_native", "dtype.element_ty"):
            assert banned not in src, f"{banned} is back in {obj.__name__}"
        body = src.split('"""')[-1].splitlines()
        arms = [
            line.strip() for line in body
            if line.strip().startswith(("if ", "elif ", "else"))
        ]
        assert not arms, f"{obj.__name__} must not branch: {arms}"


# ---- acceptance extras: graph capture, append isolation, geometry edge cases ----

def _store_into(k, v, out_loc, k_cache, v_cache, k_scale, v_scale, dim=None):
    """Store through the public entry point with caller-owned buffers.

    ``k``/``v`` are ``[T, heads * dim]`` rows (possibly slices with a wider pitch)."""
    tokens = k.shape[0]
    heads = k_scale.shape[1]
    dim = dim if dim is not None else k_cache.shape[2]
    from freetoken.kernel.triton.kv_quant import quantize_kv_to_cache

    quantize_kv_to_cache(
        k=k.reshape(tokens, heads * dim),
        v=v.reshape(tokens, heads * dim),
        out_loc=out_loc,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()


def test_empty_input_is_a_noop():
    _init_tp()
    cache = alloc_codes((4, 2, 8), DEV)
    scale = torch.full((4, 2), 7.0, dtype=KV_SCALE_DTYPE, device=DEV)
    _store_into(
        torch.empty(0, 2, 8, dtype=torch.bfloat16, device=DEV),
        torch.empty(0, 2, 8, dtype=torch.bfloat16, device=DEV),
        torch.empty(0, dtype=torch.int32, device=DEV),
        cache, cache.clone(), scale, scale.clone(),
    )
    assert (codes_to_f32(cache) == 0).all() and (scale == 7.0).all()


def test_append_leaves_the_committed_prefix_bitwise_unchanged():
    _init_tp()
    torch.manual_seed(3)
    heads, dim, slots = 2, 32, 12
    cache_k = alloc_codes((slots, heads, dim), DEV)
    cache_v = alloc_codes((slots, heads, dim), DEV)
    scale_k = torch.zeros((slots, heads), dtype=KV_SCALE_DTYPE, device=DEV)
    scale_v = torch.zeros_like(scale_k)

    first = torch.randn(4, heads, dim, dtype=torch.bfloat16, device=DEV)
    _store_into(first, first.clone(),
                torch.arange(4, dtype=torch.int32, device=DEV),
                cache_k, cache_v, scale_k, scale_v)
    snap_k = cache_k.clone()
    snap_scale = scale_k.clone()

    second = torch.randn(4, heads, dim, dtype=torch.bfloat16, device=DEV) * 10.0
    _store_into(second, second.clone(),
                torch.arange(4, 8, dtype=torch.int32, device=DEV),
                cache_k, cache_v, scale_k, scale_v)
    assert torch.equal(cache_k[:4], snap_k[:4]), "append rewrote committed codes"
    assert torch.equal(scale_k[:4], snap_scale[:4]), "append rewrote committed scales"
    assert not torch.equal(cache_k[4:8], snap_k[4:8]), "append did not write its slots"


def test_distinct_kv_row_pitches_are_honoured():
    """K and V arrive as slices of differently padded projections: each has a row
    pitch of its own, and reading both with one pitch corrupts the other row."""
    _init_tp()
    torch.manual_seed(4)
    tokens, heads, dim = 6, 2, 16
    k_wide = torch.randn(tokens, heads * dim + 5, device=DEV, dtype=torch.bfloat16)
    v_wide = torch.randn(tokens, heads * dim + 9, device=DEV, dtype=torch.bfloat16)
    k = k_wide[:, : heads * dim]
    v = v_wide[:, 9 : 9 + heads * dim]

    k_cache = alloc_codes((tokens, heads, dim), DEV)
    v_cache = alloc_codes((tokens, heads, dim), DEV)
    k_scale = torch.zeros((tokens, heads), dtype=KV_SCALE_DTYPE, device=DEV)
    v_scale = torch.zeros_like(k_scale)
    _store_into(k, v, torch.arange(tokens, dtype=torch.int32, device=DEV),
                k_cache, v_cache, k_scale, v_scale)

    for rows, cache, scale in ((k, k_cache, k_scale), (v, v_cache, v_scale)):
        f32 = rows.reshape(tokens, heads, dim).to(torch.float32)
        ref_scale = f32.abs().amax(dim=-1, keepdim=True) / FP8_MAX
        expected = _canonical_zero(_ref_codes((f32 / ref_scale).clamp(-FP8_MAX, FP8_MAX)))
        got = _canonical_zero(_as_bytes(cache).reshape(-1).to(torch.int32))
        assert torch.equal(got, expected)


def test_head_dim_below_the_launch_block_is_masked():
    """BLOCK_D covers the next power of two; the tail past head_dim must stay unwritten."""
    _init_tp()
    torch.manual_seed(5)
    tokens, heads, dim = 3, 2, 48  # 48 < BLOCK_D 64
    k = torch.randn(tokens, heads, dim, device=DEV, dtype=torch.bfloat16)
    k_cache = alloc_codes((tokens, heads, dim), DEV)
    k_scale = torch.zeros((tokens, heads), dtype=KV_SCALE_DTYPE, device=DEV)
    _store_into(k, k.clone(), torch.arange(tokens, dtype=torch.int32, device=DEV),
                k_cache, alloc_codes((tokens, heads, dim), DEV), k_scale,
                torch.zeros_like(k_scale))
    f32 = k.to(torch.float32)
    ref_scale = f32.abs().amax(dim=-1, keepdim=True) / FP8_MAX
    expected = _canonical_zero(_ref_codes((f32 / ref_scale).clamp(-FP8_MAX, FP8_MAX)))
    got = _canonical_zero(_as_bytes(k_cache).reshape(-1).to(torch.int32))
    assert torch.equal(got, expected)


def test_saturation_clamps_at_the_e4m3_max():
    _init_tp()
    rows = torch.tensor([[[1000.0, -1000.0, 3.0, 0.5]] * 1], dtype=torch.bfloat16, device=DEV)
    k_cache, _, k_scale, _ = _store(rows, rows.clone())
    codes = _as_bytes(k_cache)[0, 0]
    assert codes[0] == CODE_448 and codes[1] == CODE_448 | 0x80, (codes[:2].tolist(),)
    deq = codes_to_f32(k_cache) * k_scale.unsqueeze(-1)
    torch.testing.assert_close(
        deq[0, 0, 0], torch.tensor(1000.0, device=DEV), rtol=1e-3, atol=0
    )
    torch.testing.assert_close(
        deq[0, 0, 1], torch.tensor(-1000.0, device=DEV), rtol=1e-3, atol=0
    )


def test_ties_round_to_the_even_code():
    """The midpoint of two adjacent grid values is stored as the even code (RNE).

    The row carries a 448.0 anchor, so its scale is exactly 1.0 and each byte IS the
    encoder's answer for the value placed there. Midpoints of e4m3 values are dyadic
    with few enough bits to be exact in bf16, so no input rounding blurs the tie.
    """
    _init_tp()
    pairs = sorted(E4M3_VALUES.items())
    ties, expected = [], []
    for (ca, va), (cb, vb) in zip(pairs, pairs[1:]):
        if vb <= va:  # -0.0/+0.0 duplicate, or the subnormal sleeve; nothing to tie
            continue
        even = ca if ca % 2 == 0 else (cb if cb % 2 == 0 else None)
        if even is None or (ca >> 7) != (cb >> 7):
            continue  # sign changes never tie; parity must map to the code's LSB
        ties.append((va + vb) / 2.0)
        expected.append(even)
    dim = len(ties) + 1
    row = torch.tensor([[[FP8_MAX] + ties]], dtype=torch.bfloat16, device=DEV)
    k_cache, _, k_scale, _ = _store(row, row.clone())
    assert torch.equal(k_scale, torch.ones_like(k_scale))
    got = _as_bytes(k_cache)[0, 0, 1:].to(torch.int32)
    want = torch.tensor(expected, dtype=torch.int32, device=DEV)
    bad = got != want
    assert not bad.any().item(), (
        f"{int(bad.sum())} of {len(expected)} ties resolved wrong; "
        f"first at {int(bad.nonzero()[0])}: got {int(got[bad][0]):#x} want {int(want[bad][0]):#x}"
    )


def test_dummy_page_decodes_to_exact_zero():
    """Unwritten slots decode to exactly 0.0 and hold no NaN pattern (0x7F/0xFF)."""
    dummy = alloc_codes((2, 4, 8), DEV)
    assert (codes_to_f32(dummy) == 0).all()
    raw = _as_bytes(dummy).reshape(-1)
    assert not ((raw == 0x7F) | (raw == 0xFF)).any()


def test_capture_and_replay_with_varying_out_loc():
    """The scatter must be CUDA-graph capturable, with slot ids supplied per replay.

    The slot tensor is filled by a device copy inside the graph, so both replays go
    through one captured launch; the second replay must land on different slots and
    leave the first replay's rows alone.
    """
    from freetoken.kernel.triton.kv_quant import quantize_kv_to_cache

    _init_tp()
    torch.manual_seed(6)
    tokens, heads, dim, slots = 4, 2, 64, 16
    rows = torch.randn(tokens, heads, dim, device=DEV, dtype=torch.bfloat16)
    cache_k = alloc_codes((slots, heads, dim), DEV)
    cache_v = alloc_codes((slots, heads, dim), DEV)
    scale_k = torch.zeros((slots, heads), dtype=KV_SCALE_DTYPE, device=DEV)
    scale_v = torch.zeros_like(scale_k)
    out_loc = torch.zeros(tokens, dtype=torch.int32, device=DEV)
    fill = torch.arange(tokens, dtype=torch.int32, device=DEV)
    src = torch.zeros_like(fill)  # slot ids the graph copies into out_loc each replay

    def launch():
        quantize_kv_to_cache(
            k=rows.reshape(tokens, -1), v=rows.reshape(tokens, -1), out_loc=out_loc,
            k_cache=cache_k, v_cache=cache_v, k_scale=scale_k, v_scale=scale_v,
        )

    launch()  # JIT compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_loc.copy_(src)
        launch()

    src.copy_(fill)
    graph.replay()
    torch.cuda.synchronize()
    first = cache_k[:tokens].clone()
    first_scale = scale_k[:tokens].clone()

    src.copy_(fill + 8)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(cache_k[:tokens], first), "replay rewrote the first slot range"
    assert torch.equal(scale_k[:tokens], first_scale)
    assert torch.equal(cache_k[8 : 8 + tokens], first), "replay missed the new slots"
    assert torch.equal(scale_k[8 : 8 + tokens], first_scale)
    expected = (
        codes_to_f32(first) * first_scale.unsqueeze(-1)
    )
    f32 = rows.to(torch.float32)
    ref_scale = f32.abs().amax(dim=-1, keepdim=True) / FP8_MAX
    assert torch.all((expected - f32).abs() <= 0.08 * f32.abs().amax(dim=-1, keepdim=True))
    assert torch.all(ref_scale > 0)


def test_tiny_rows_use_the_amax_floor_and_still_round_trip():
    """Values far below the e4m3 grid are NOT collapsed to zero: the 1e-10 amax floor
    (the activation quant's own floor) keeps the row scale usable and the ratio
    encodes the value's magnitude relative to it."""
    _init_tp()
    row = torch.full((1, 1, 16), 1e-12, dtype=torch.bfloat16, device=DEV)
    k_cache, _, k_scale, _ = _store(row, row.clone())
    torch.testing.assert_close(
        k_scale, torch.full_like(k_scale, 1e-10 / FP8_MAX), rtol=1e-6, atol=0
    )
    assert (codes_to_f32(k_cache) != 0).any(), "tiny rows must not collapse to zero"
    deq = codes_to_f32(k_cache) * k_scale.unsqueeze(-1)
    torch.testing.assert_close(deq, row.to(torch.float32), rtol=0.08, atol=1e-18)


def test_non_finite_rows_are_not_sanitized():
    """Documented contract: NaN/Inf rows are NOT cleaned up, and the observed outcome
    is pinned here so a "cleanup" cannot land silently.

    - A NaN row's amax is the largest finite magnitude (tl.maximum's maxnum semantics
      ignore the NaN), so every value clamps to +-448 and the pooled read is finite
      but wrong: ~the finite amax on every element.
    - An Inf row's amax is Inf, so its scale is Inf; finite entries then read as
      0 * Inf = NaN and the Inf entries as -448 * Inf = -Inf.
    Attention backends guarantee finite K/V; the pool must not be fed non-finite rows.
    """
    _init_tp()
    nan_row = torch.tensor([[[float("nan")] * 4 + [1.0] * 4]], dtype=torch.bfloat16, device=DEV)
    k_cache, _, k_scale, _ = _store(nan_row, nan_row.clone())
    # tl.maximum's maxnum semantics ignore the NaN, so the scale is the finite
    # entries' amax over 448 and each NaN clamps to +-448: finite, ~1.0-magnitude junk.
    torch.testing.assert_close(
        k_scale, torch.full_like(k_scale, 1.0 / FP8_MAX), rtol=1e-6, atol=0
    )
    read = codes_to_f32(k_cache) * k_scale.unsqueeze(-1)
    assert torch.isfinite(read).all()

    inf_row = torch.tensor([[[float("inf")] * 4 + [1.0] * 4]], dtype=torch.bfloat16, device=DEV)
    k_cache, _, k_scale, _ = _store(inf_row, inf_row.clone())
    assert torch.isinf(k_scale).all()
    read = codes_to_f32(k_cache) * k_scale.unsqueeze(-1)
    assert not torch.isfinite(read).all(), "an Inf row must read back non-finite"
