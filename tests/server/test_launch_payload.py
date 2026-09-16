"""Worker-group spawn payload: a respawn must survive the live args becoming unpicklable.

The DP watchdog respawns a dead engine group through ``handle.restart``; in production the live
``ServerArgs`` stopped pickling after the engine had been serving (a quant Matcher closure:
``AttributeError: Can't get local object 'name_set.<locals>.hit'``), so every respawn failed and
the engine stayed dead until a manual service restart. ``_spawn_payload`` freezes the payload at
the first spawn and reuses it for every respawn.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import pytest

from freetoken.server.launch import _spawn_payload


@dataclass
class _Args:
    model_path: str = "/models/x"
    dp_index: int = 1
    server_port: int = 1919


def test_first_spawn_freezes_the_payload() -> None:
    args = _Args()
    frozen, spawn_args = _spawn_payload(args, None)
    assert isinstance(frozen, bytes)
    assert spawn_args is args  # the first spawn already holds a picklable object
    assert pickle.loads(frozen).dp_index == 1


def test_respawn_reuses_the_frozen_payload_after_drift() -> None:
    args = _Args()
    frozen, _ = _spawn_payload(args, None)

    def hit(name: str) -> bool:  # a closure: __qualname__ contains "<locals>", so it never pickles
        return True

    args.matcher = hit  # the exact drift shape that killed the production respawn
    with pytest.raises(Exception):
        pickle.dumps(args)

    frozen2, spawn_args = _spawn_payload(args, frozen)
    assert frozen2 == frozen
    assert spawn_args.dp_index == 1
    assert not hasattr(spawn_args, "matcher")
    pickle.dumps(spawn_args)  # what the watchdog is about to hand to mp.Process


def test_respawn_keeps_the_startup_values() -> None:
    args = _Args()
    frozen, _ = _spawn_payload(args, None)
    args.server_port = 1234  # a live mutation must not leak into a reborn group
    _, spawn_args = _spawn_payload(args, frozen)
    assert spawn_args.server_port == 1919
