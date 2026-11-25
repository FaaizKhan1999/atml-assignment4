"""Paired bootstrap confidence intervals (percentile) and a Pearson chi-square GOF helper.

Resampling unit is configurable:
  * unit="example"/"prompt": each entry of the per-unit arrays is one resampling unit;
  * grouped: pass `groups` (one label per row) to resample whole groups (e.g. whole evaluation windows
    or prompts with several rows each) with replacement.
Defaults: 2,000 resamples, seed 6304, 95% percentile interval.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 6304


def _group_index(n: int, groups: Sequence | None) -> list[np.ndarray]:
    if groups is None:
        return [np.array([i]) for i in range(n)]
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError("groups must have one label per row")
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    return np.split(order, np.cumsum(counts)[:-1])


def _resample_indices(blocks: list[np.ndarray], n_resamples: int, rng: np.random.Generator):
    g = len(blocks)
    for _ in range(n_resamples):
        pick = rng.integers(0, g, size=g)
        yield np.concatenate([blocks[i] for i in pick])


def _summary(point, reps, alpha, n_units, n_resamples, unit, seed):
    reps = np.asarray(reps, dtype=float)
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi), "level": 1 - alpha,
            "n": int(n_units), "n_resamples": int(n_resamples), "unit": unit, "seed": int(seed),
            "method": "percentile"}


def bootstrap_ci(values, *, stat: Callable[[np.ndarray], float] = np.mean, groups=None,
                 n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED, alpha: float = 0.05,
                 unit: str = "example") -> dict:
    """Percentile bootstrap CI of stat(values)."""
    v = np.asarray(values, dtype=float)
    blocks = _group_index(len(v), groups)
    rng = np.random.default_rng(seed)
    reps = [stat(v[idx]) for idx in _resample_indices(blocks, n_resamples, rng)]
    return _summary(stat(v), reps, alpha, len(blocks), n_resamples, unit, seed)


def paired_bootstrap_diff(a, b, *, stat: Callable[[np.ndarray], float] = np.mean, groups=None,
                          n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED,
                          alpha: float = 0.05, unit: str = "example") -> dict:
    """Paired percentile bootstrap CI for stat(a) - stat(b); a[i] and b[i] are the same unit/row."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have identical shape")
    blocks = _group_index(len(a), groups)
    rng = np.random.default_rng(seed)
    reps = [stat(a[idx]) - stat(b[idx]) for idx in _resample_indices(blocks, n_resamples, rng)]
    out = _summary(stat(a) - stat(b), reps, alpha, len(blocks), n_resamples, unit, seed)
    out.update({"stat_a": float(stat(a)), "stat_b": float(stat(b))})
    return out


def pooled_ppl(nll_sums, token_counts) -> float:
    """exp(sum NLL / sum tokens) over windows."""
    return float(np.exp(np.sum(nll_sums) / np.sum(token_counts)))


def pooled_nll_bootstrap(nll_sums_a, token_counts, nll_sums_b=None, *, n_resamples: int = DEFAULT_RESAMPLES,
                         seed: int = DEFAULT_SEED, alpha: float = 0.05) -> dict:
    """Perplexity bootstrap over whole evaluation windows.

    Each resample draws windows with replacement, recomputes pooled token NLL = sum(nll)/sum(tokens), then
    exponentiates. With nll_sums_b, returns the paired difference PPL_a - PPL_b (same windows / token counts,
    which must be identical across models) and also the pooled mean-NLL difference.
    """
    na = np.asarray(nll_sums_a, dtype=float)
    nt = np.asarray(token_counts, dtype=float)
    if na.shape != nt.shape:
        raise ValueError("one token count per window required")
    nb = None if nll_sums_b is None else np.asarray(nll_sums_b, dtype=float)
    if nb is not None and nb.shape != na.shape:
        raise ValueError("paired windows must match")
    rng = np.random.default_rng(seed)
    w = len(na)
    ppl_reps, nll_reps = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, w, size=w)
        t = nt[idx].sum()
        ma = na[idx].sum() / t
        if nb is None:
            ppl_reps.append(np.exp(ma))
            nll_reps.append(ma)
        else:
            mb = nb[idx].sum() / t
            ppl_reps.append(np.exp(ma) - np.exp(mb))
            nll_reps.append(ma - mb)
    T = nt.sum()
    mean_a = na.sum() / T
    if nb is None:
        ppl_pt, nll_pt = np.exp(mean_a), mean_a
    else:
        mean_b = nb.sum() / T
        ppl_pt, nll_pt = np.exp(mean_a) - np.exp(mean_b), mean_a - mean_b
    out = {"ppl": _summary(ppl_pt, ppl_reps, alpha, w, n_resamples, "window", seed),
           "mean_nll": _summary(nll_pt, nll_reps, alpha, w, n_resamples, "window", seed),
           "scored_tokens": int(T), "paired": nb is not None}
    return out


def chisquare_gof(observed, expected_probs, *, min_expected: float = 10.0, ddof: int = 0) -> dict:
    """Pearson chi-square goodness-of-fit of observed counts against expected probabilities.

    Raises if any expected count is below `min_expected` (the assignment requires all >= 10).
    """
    from scipy import stats

    obs = np.asarray(observed, dtype=float)
    p = np.asarray(expected_probs, dtype=float)
    if obs.shape != p.shape:
        raise ValueError("shape mismatch")
    if not np.isclose(p.sum(), 1.0, atol=1e-9):
        raise ValueError(f"expected probabilities sum to {p.sum()}")
    exp = p * obs.sum()
    if exp.min() < min_expected:
        raise ValueError(f"min expected count {exp.min():.2f} < {min_expected}")
    res = stats.chisquare(obs, exp, ddof=ddof)
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "dof": int(len(obs) - 1 - ddof),
            "n": int(obs.sum()), "min_expected": float(exp.min()), "categories": int(len(obs))}
