"""--kv-cache-dtype parsing and the bf16-compatible default."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from freetoken.server.args import parse_args

ANON_PATH = "/models/anon"


class _Config:
    def __init__(self, data: dict) -> None:
        self._data = data

    def to_dict(self) -> dict:
        return self._data


def _parse(extra: list[str]):
    config = _Config({"architectures": ["Qwen4ExpForConditionalGeneration"],
                      "torch_dtype": "bfloat16"})
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: config):
        return parse_args(["--model", ANON_PATH, *extra])[0]


def test_kv_cache_dtype_defaults_to_the_compute_dtype():
    assert _parse([]).kv_quant == "none"


def test_kv_cache_dtype_spellings_parse():
    assert _parse(["--kv-cache-dtype", "auto"]).kv_quant == "auto"
    assert _parse(["--kv-cache-dtype", "bf16"]).kv_quant == "bf16"
    assert _parse(["--kv-cache-dtype", "fp8"]).kv_quant == "fp8"
    assert _parse(["--kv-cache-dtype", "nvfp4"]).kv_quant == "nvfp4"


def test_kv_cache_dtype_rejects_unknown_spellings():
    with pytest.raises(SystemExit):
        _parse(["--kv-cache-dtype", "q8"])
