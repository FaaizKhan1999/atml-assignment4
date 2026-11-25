"""P3: OWL-style nonuniform block sparsity on top of the unchanged Wanda per-row rule (pre-registered design)."""
from __future__ import annotations

import math

import torch

from .model import ELIGIBLE, eligible_projections
from .prune import row_mask_k, wanda_score

LO, HI = 0.10, 0.85
GRID_M = (3, 5, 7, 10)
GRID_LAMBDA = (0.02, 0.05, 0.08, 0.12, 0.16, 0.20)


@torch.no_grad()
def block_outlier_ratios(model, xnorms: dict, Ms=GRID_M) -> dict[int, list[float]]:
    """D_l(M) = #{Wanda > M*mean_projection} summed over the block's 7 projections / eligible params of block l."""
    L = model.config.num_hidden_layers
    cnt = {M: [0] * L for M in Ms}
    numel = [0] * L
    for p in eligible_projections(model):
        s = wanda_score(p.module.weight, xnorms[p.qualname])
        mu = s.mean()
        for M in Ms:
            cnt[M][p.layer] += int((s > M * mu).sum())
        numel[p.layer] += s.numel()
    return {M: [c / n for c, n in zip(cnt[M], numel)] for M in Ms}


def owl_allocation(D: list[float], S: float, lam: float, weights=None, iters: int = 200) -> tuple[list[float], float]:
    """s_l = clip(S + lam - 2 lam (D_l - minD)/(maxD - minD) + c, LO, HI) with c by bisection so that the
    parameter-weighted mean equals S."""
    n = len(D)
    w = [1.0] * n if weights is None else list(weights)
    lo_d, hi_d = min(D), max(D)
    if hi_d == lo_d:
        raw = [S] * n
    else:
        raw = [S + lam - 2 * lam * (d - lo_d) / (hi_d - lo_d) for d in D]

    def mean_at(c):
        return sum(wi * min(HI, max(LO, r + c)) for wi, r in zip(w, raw)) / sum(w)

    a, b = -1.0, 1.0
    for _ in range(iters):
        m = (a + b) / 2
        if mean_at(m) < S:
            a = m
        else:
            b = m
    c = (a + b) / 2
    return [min(HI, max(LO, r + c)) for r in raw], c


def row_counts(proj_shapes: list[tuple[int, int, int]], s_blocks: list[float], target_zeros: int
               ) -> tuple[list[torch.Tensor], dict]:
    """proj_shapes: (layer, d_out, d_in) in eligible order. Base k = round(s_l*d_in) per row; then adjust one
    row at a time by sign(Delta) in (block, projection, row) order until the total equals target_zeros."""
    ks = [torch.full((d_out,), round(s_blocks[l] * d_in), dtype=torch.long) for l, d_out, d_in in proj_shapes]
    total = sum(int(k.sum()) for k in ks)
    delta = target_zeros - total
    rows = sum(d_out for _, d_out, _ in proj_shapes)
    if abs(delta) > rows:
        raise AssertionError(f"|Delta|={abs(delta)} exceeds row count {rows}")
    sign = 1 if delta > 0 else -1
    for k, (l, d_out, d_in) in zip(ks, proj_shapes):
        if delta == 0:
            break
        n = min(abs(delta), d_out)
        k[:n] += sign
        delta -= sign * n
    assert delta == 0
    for k, (_, _, d_in) in zip(ks, proj_shapes):
        assert int(k.min()) >= 0 and int(k.max()) <= d_in
    final = sum(int(k.sum()) for k in ks)
    assert final == target_zeros, (final, target_zeros)
    L = max(l for l, _, _ in proj_shapes) + 1
    per_block = [0] * L
    for k, (l, _, _) in zip(ks, proj_shapes):
        per_block[l] += int(k.sum())
    return ks, {"base_total": total, "adjusted_rows": abs(target_zeros - total), "per_block_zeros": per_block}


def target_zero_count(model, S: float) -> int:
    n = sum(p.module.weight.numel() for p in eligible_projections(model))
    z = S * n
    if abs(z - round(z)) > 1e-6:
        raise AssertionError("S * #eligible is not an integer")
    return int(round(z))


@torch.no_grad()
def apply_p3(model, xnorms: dict, D: list[float], S: float, lam: float) -> dict:
    projs = eligible_projections(model)
    L = model.config.num_hidden_layers
    block_numel = [0] * L
    for p in projs:
        block_numel[p.layer] += p.module.weight.numel()
    s_blocks, c = owl_allocation(D, S, lam, weights=block_numel)
    target = target_zero_count(model, S)
    ks, rinfo = row_counts([(p.layer, p.shape[0], p.shape[1]) for p in projs], s_blocks, target)
    pruned = 0
    for p, k in zip(projs, ks):
        m = row_mask_k(wanda_score(p.module.weight, xnorms[p.qualname]), k)
        pruned += int(m.sum())
        p.module.weight.masked_fill_(m, 0)
    assert pruned == target
    return {"S": S, "lambda": lam, "s_blocks": s_blocks, "shift_c": c, "D": D, "target_zeros": target,
            "mask_pruned": pruned, "mask_pruned_fraction": pruned / sum(block_numel),
            "weighted_mean_s": sum(s * n for s, n in zip(s_blocks, block_numel)) / sum(block_numel), **rinfo,
            "eligible_order": list(ELIGIBLE)}


@torch.no_grad()
def dev_nll(model, recs: list[dict], device) -> dict:
    """C4 dev manifest: each 512-token sequence is one window, targets 1..511 (harness PPL code)."""
    from .ppl import window_nll

    s, n = 0.0, 0
    for r in recs:
        ids = r["token_ids"]
        s += window_nll(model, ids, 1, len(ids), device)
        n += len(ids) - 1
    return {"sum_nll": s, "tokens": n, "mean_nll": s / n, "ppl": math.exp(s / n)}
