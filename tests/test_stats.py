import numpy as np
import pytest

from atml4.common.stats import (bootstrap_ci, chisquare_gof, paired_bootstrap_diff, pooled_nll_bootstrap,
                                pooled_ppl)


def test_bootstrap_known_case_mean():
    # Bernoulli(0.7) sample, n=2000: percentile CI should be close to normal approx +/-1.96*sqrt(p(1-p)/n)
    rng = np.random.default_rng(0)
    x = (rng.random(2000) < 0.7).astype(float)
    r = bootstrap_ci(x)
    se = np.sqrt(x.mean() * (1 - x.mean()) / len(x))
    assert r["point"] == pytest.approx(x.mean())
    assert r["ci_low"] == pytest.approx(x.mean() - 1.96 * se, abs=0.006)
    assert r["ci_high"] == pytest.approx(x.mean() + 1.96 * se, abs=0.006)
    assert r["n"] == 2000 and r["n_resamples"] == 2000 and r["seed"] == 6304


def test_constant_degenerate():
    r = paired_bootstrap_diff(np.ones(50), np.zeros(50))
    assert r["point"] == r["ci_low"] == r["ci_high"] == 1.0


def test_paired_identical_is_zero_and_deterministic():
    x = np.random.default_rng(1).normal(size=100)
    r = paired_bootstrap_diff(x, x)
    assert r["point"] == r["ci_low"] == r["ci_high"] == 0.0
    y = x + np.random.default_rng(2).normal(scale=0.1, size=100)
    assert paired_bootstrap_diff(x, y) == paired_bootstrap_diff(x, y)


def test_paired_narrower_than_unpaired():
    rng = np.random.default_rng(3)
    base = rng.normal(size=500) * 5
    a, b = base + 0.1, base + rng.normal(scale=0.05, size=500)
    paired = paired_bootstrap_diff(a, b)
    assert paired["ci_high"] - paired["ci_low"] < 0.05


def test_grouped_units():
    rows = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    groups = ["p0", "p0", "p1", "p1", "p2", "p2"]
    r = bootstrap_ci(rows, groups=groups, unit="prompt")
    assert r["n"] == 3 and r["unit"] == "prompt"


def test_pooled_nll_bootstrap():
    rng = np.random.default_rng(4)
    tok = rng.integers(100, 512, size=120).astype(float)
    nll = tok * rng.normal(2.5, 0.2, size=120)
    r = pooled_nll_bootstrap(nll, tok)
    expect = np.exp(nll.sum() / tok.sum())
    assert r["ppl"]["point"] == pytest.approx(expect)
    assert r["ppl"]["point"] == pytest.approx(pooled_ppl(nll, tok))
    assert r["ppl"]["ci_low"] < expect < r["ppl"]["ci_high"]
    assert r["scored_tokens"] == int(tok.sum()) and r["ppl"]["unit"] == "window"
    # pooled is not the mean of per-window ppl
    assert not np.isclose(expect, np.mean(np.exp(nll / tok)))
    # paired: identical models -> exactly zero
    rp = pooled_nll_bootstrap(nll, tok, nll)
    assert rp["ppl"]["point"] == rp["ppl"]["ci_low"] == rp["ppl"]["ci_high"] == 0.0
    # paired shift of +0.1 nat/token everywhere -> mean_nll difference exactly 0.1
    rs = pooled_nll_bootstrap(nll + 0.1 * tok, tok, nll)
    assert rs["mean_nll"]["point"] == pytest.approx(0.1)
    assert rs["mean_nll"]["ci_low"] == pytest.approx(0.1) and rs["mean_nll"]["ci_high"] == pytest.approx(0.1)


def test_chisquare():
    p = np.array([0.2, 0.3, 0.5])
    rng = np.random.default_rng(6304)
    obs = np.bincount(rng.choice(3, size=100_000, p=p), minlength=3)
    assert chisquare_gof(obs, p)["p_value"] > 0.01
    assert chisquare_gof(obs, [0.25, 0.3, 0.45])["p_value"] < 0.01
    with pytest.raises(ValueError):
        chisquare_gof([1, 2, 3], [0.98, 0.01, 0.01])
