"""--lookup-draft / --lookup-ngram parsing and the one-drafter-per-run rule."""

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


def test_lookup_defaults_are_off():
    args = _parse([])
    assert args.lookup_draft == 0 and args.lookup_ngram == 6


def test_lookup_flags_parse():
    args = _parse(["--lookup-draft", "4", "--lookup-ngram", "5", "--lookup-min-ngram", "3"])
    assert (args.lookup_draft, args.lookup_ngram, args.lookup_min_ngram) == (4, 5, 3)


def test_lookup_and_mtp_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        _parse(["--lookup-draft", "2", "--mtp-depth", "2"])
