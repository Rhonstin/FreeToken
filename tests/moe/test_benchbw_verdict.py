"""(#37) `ft bench bw` must not present a coin flip as a fact.

The hybrid-vs-offload verdict is a bandwidth ratio against a threshold. Some formats
measure stably; others swing far enough between runs to land on both sides of it.
`verdict` bounds the ratio by the extremes actually observed and withholds the call
(resolving to offload) when that interval straddles the threshold.
"""

from freetoken.moe.benchbw import verdict


def test_clearly_above_threshold_is_hybrid():
    pick, confident, (lo, hi) = verdict([96.0, 97.0, 95.5], [25.0, 25.1, 25.0], 2.0)
    assert (pick, confident) == ("hybrid", True)
    assert lo > 2.0 and hi > lo


def test_clearly_below_threshold_is_offload():
    pick, confident, _ = verdict([40.0, 41.0, 39.5], [25.0, 25.1, 25.0], 2.0)
    assert (pick, confident) == ("offload", True)


def test_straddling_the_threshold_is_withheld():
    pick, confident, (lo, hi) = verdict([42.2, 50.2, 60.0], [25.2, 25.0, 25.1], 2.0)
    assert lo < 2.0 < hi, (lo, hi)
    assert confident is False
    assert pick == "offload", "an undecided measurement must fall back to the safe backend"


def test_a_single_run_still_decides():
    assert verdict([96.0], [25.0], 2.0)[:2] == ("hybrid", True)
    assert verdict([40.0], [25.0], 2.0)[:2] == ("offload", True)


def test_exactly_at_the_threshold_is_not_hybrid():
    pick, confident, _ = verdict([50.0], [25.0], 2.0)
    assert (pick, confident) == ("offload", True)
