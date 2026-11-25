import time

import pytest

from atml4.common.timing import WallTimer, time_cpu, time_fn


def test_time_cpu():
    r = time_cpu(lambda: time.sleep(0.002), reps=5, warmup=1)
    assert len(r.samples) == r.reps == 5 and r.unit == "ms" and r.device == "cpu"
    assert all(s >= 1.9 for s in r.samples)
    assert r.min <= r.median <= r.max
    d = r.to_dict()
    assert d["samples"] == r.samples and d["warmup"] == 1


def test_min_reps_enforced():
    with pytest.raises(ValueError):
        time_cpu(lambda: None, reps=3)
    with pytest.raises(ValueError):
        time_cpu(lambda: None, reps=5, min_reps=10)


def test_time_fn_and_walltimer():
    r = time_fn(lambda: sum(range(1000)), reps=5)
    assert len(r.samples) == 5
    with WallTimer() as t:
        time.sleep(0.003)
    assert t.ms >= 2.9


def test_setup_not_timed():
    calls = []
    r = time_cpu(lambda s: calls.append(s), reps=5, warmup=1, setup=lambda: (time.sleep(0.01), "st")[1])
    assert calls == ["st"] * 6
    assert max(r.samples) < 5  # the 10 ms setup sleep is outside the timed region
