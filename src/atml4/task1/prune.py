"""Unstructured per-row pruning (magnitude / Wanda), 2:4 masks, and per-projection score diagnostics (RQ1)."""
from __future__ import annotations

import numpy as np
import torch

QUANTILES = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0]
OWL_M = [3, 5, 7, 10]
SPEARMAN_SAMPLE = 200_000


def magnitude_score(W: torch.Tensor) -> torch.Tensor:
    return W.float().abs()


def wanda_score(W: torch.Tensor, xnorm: torch.Tensor) -> torch.Tensor:
    """|W_ij| * ||X_j||_2 (xnorm: per input channel L2 norm over all calibration tokens)."""
    return W.float().abs() * xnorm.float().to(W.device)[None, :]


def k_per_row(sparsity: float, d_in: int) -> int:
    k = sparsity * d_in
    if abs(k - round(k)) > 1e-9:
        raise AssertionError(f"s*d_in = {k} is not an integer")
    return int(round(k))


def row_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Boolean prune-mask (True = zeroed): per row the k lowest scores; ties -> lower column index pruned first."""
    d_out, d_in = score.shape
    k = k_per_row(sparsity, d_in)
    idx = torch.sort(score, dim=1, stable=True).indices[:, :k]
    m = torch.zeros_like(score, dtype=torch.bool)
    m.scatter_(1, idx, True)
    return m


def two_four_mask(score: torch.Tensor) -> torch.Tensor:
    """2:4 prune-mask: in each group of 4 contiguous input columns per row, zero the 2 lowest scores
    (ties -> lower column index pruned first)."""
    d_out, d_in = score.shape
    if d_in % 4:
        raise ValueError("d_in must be divisible by 4")
    g = score.view(d_out, d_in // 4, 4)
    idx = torch.sort(g, dim=-1, stable=True).indices[..., :2]
    m = torch.zeros_like(g, dtype=torch.bool)
    m.scatter_(-1, idx, True)
    return m.view(d_out, d_in)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / torch.sqrt((ra * ra).sum() * (rb * rb).sum()))


def score_diagnostics(W: torch.Tensor, xnorm: torch.Tensor, sparsities=(0.25, 0.5, 0.75), seed: int = 6304,
                      layer_index: int = 0) -> dict:
    """Per-projection diagnostics: score quantiles, OWL outlier ratios, Spearman(|W|, Wanda), mask Jaccard."""
    mag = magnitude_score(W)
    wan = wanda_score(W, xnorm)
    q = torch.tensor(QUANTILES, device=W.device)
    flat_m, flat_w = mag.flatten(), wan.flatten()

    def quant(x):
        if x.numel() > 16_000_000:
            raise ValueError("matrix too large for torch.quantile")
        return [float(v) for v in torch.quantile(x, q)]

    mean_w = float(flat_w.mean())
    g = torch.Generator(device="cpu").manual_seed(seed + layer_index)
    n = flat_w.numel()
    idx = torch.randperm(n, generator=g)[: min(SPEARMAN_SAMPLE, n)].to(W.device)
    out = {"magnitude_quantiles": quant(flat_m), "wanda_quantiles": quant(flat_w), "quantile_levels": QUANTILES,
           "wanda_mean": mean_w, "magnitude_mean": float(flat_m.mean()),
           "owl_outlier_ratio": {str(M): float((flat_w > M * mean_w).double().mean()) for M in OWL_M},
           "spearman_absW_wanda": _spearman(flat_m[idx], flat_w[idx]),
           "spearman_sample": int(idx.numel()), "spearman_seed": seed + layer_index,
           "xnorm_quantiles": quant(xnorm.float().to(W.device)), "jaccard_pruned_mag_vs_wanda": {}}
    for s in sparsities:
        a, b = row_mask(mag, s), row_mask(wan, s)
        inter = int((a & b).sum())
        union = int((a | b).sum())
        out["jaccard_pruned_mag_vs_wanda"][str(s)] = inter / union
    return out


def xnorms_from_stats(stats: dict) -> dict[str, torch.Tensor]:
    return {q: torch.from_numpy(np.sqrt(v["sumsq"].astype(np.float64))).float() for q, v in stats.items()}
