"""Hardware-profile fingerprint matcher.

The reader must refuse a profile from another machine even when the GPU name matches.
These are pure (no torch, no GPU): they pin the schema gate, the hard/optional field
rules and the bandwidth sanity checks.
"""

import copy
import math

from freetoken.moe.bench_profile import (
    profile_bandwidths_valid,
    profile_compatible,
    read_machine_fingerprint,
)

FP = {
    "schema_version": 5,
    "cpu": {"family": "6", "model": "85", "stepping": "4", "isa": "avx512", "threads": 6,
            "sku": "Intel(R) Core(TM) i7-7800X CPU @ 3.50GHz"},
    "ram": {"total_bytes": 104411424 * 1024, "observed_channels": 4, "configured_speed_mts": 2133},
    "pcie": {"bdf": "00000000:65:00.0", "gen_max": 3, "width_max": 16},
    "gpu": {"name": "NVIDIA GeForce RTX 3090", "sm": "8.6"},
}


def test_compatible_when_identical():
    ok, reason = profile_compatible(FP, copy.deepcopy(FP))
    assert ok, reason


def test_same_gpu_different_cpu_rejected():
    cur = copy.deepcopy(FP)
    cur["cpu"]["model"] = "158"  # a different i7
    ok, reason = profile_compatible(FP, cur)
    assert not ok and "cpu.model" in reason


def test_changed_threads_rejected():
    cur = copy.deepcopy(FP)
    cur["cpu"]["threads"] = 4
    ok, reason = profile_compatible(FP, cur)
    assert not ok and "threads" in reason


def test_same_name_different_bdf_or_link_rejected():
    for field in ("bdf", "gen_max", "width_max"):
        cur = copy.deepcopy(FP)
        cur["pcie"][field] = 99 if field != "bdf" else "00000000:17:00.0"
        ok, reason = profile_compatible(FP, cur)
        assert not ok and field in reason, (field, reason)


def test_missing_required_field_is_unverified():
    stored = copy.deepcopy(FP)
    del stored["cpu"]["model"]
    ok, reason = profile_compatible(stored, copy.deepcopy(FP))
    assert not ok and "unverified" in reason


def test_unknown_schema_rejected():
    for bad in (4, None, "5"):
        stored = copy.deepcopy(FP)
        stored["schema_version"] = bad
        ok, reason = profile_compatible(stored, copy.deepcopy(FP))
        assert not ok and "schema_version" in reason


def test_optional_field_mismatch_when_both_present():
    cur = copy.deepcopy(FP)
    cur["ram"]["configured_speed_mts"] = 2666
    ok, reason = profile_compatible(FP, cur)
    assert not ok and "configured_speed_mts" in reason


def test_optional_field_absent_is_skipped():
    stored = copy.deepcopy(FP)
    stored["ram"]["configured_speed_mts"] = None
    cur = copy.deepcopy(FP)
    cur["ram"]["configured_speed_mts"] = 2133
    ok, reason = profile_compatible(stored, cur)
    assert ok, reason


def test_bandwidth_sanity():
    good = {"dtype_kernels": {"nvfp4": {"cpu_moe_gbs": 30.0, "pcie_gather_gbs": 12.0}}}
    assert profile_bandwidths_valid(good)[0]
    for bad in (0.0, -1.0, math.nan, math.inf):
        prof = {"dtype_kernels": {"nvfp4": {"cpu_moe_gbs": bad, "pcie_gather_gbs": 12.0}}}
        ok, reason = profile_bandwidths_valid(prof)
        assert not ok and "cpu_moe_gbs" in reason


def test_read_machine_fingerprint_shape():
    fp = read_machine_fingerprint(gpu_name="FAKE GPU", threads=6)
    assert fp["schema_version"] == 5
    for sec in ("cpu", "ram", "pcie", "gpu", "runtime", "expert"):
        assert sec in fp
    assert fp["cpu"]["threads"] == 6
    # no hostname / serial in the fingerprint
    assert "host" not in fp and "serial" not in repr(fp).lower()
