"""Q3: per-block bit allocation by isolated dev-NLL probes + exact knapsack (pre-registered design)."""
from __future__ import annotations

from .quant import packed_nbytes

BITS = (2, 3, 4, 8)


def block_packed_bytes(shapes: list[tuple[int, int]], b: int) -> int:
    return sum(packed_nbytes(d_out, d_in, b) for d_out, d_in in shapes)


def knapsack(costs: list[dict[int, float]], weights: list[dict[int, int]], budget: int) -> dict:
    """Exact: minimise sum costs[l][b_l] s.t. sum weights[l][b_l] <= budget (integer bytes).
    Ties: lower cost, then fewer bytes, then lexicographically larger bit vector."""
    states: dict[int, tuple[float, tuple]] = {0: (0.0, ())}
    for l in range(len(costs)):
        nxt: dict[int, tuple[float, tuple]] = {}
        for w0, (c0, bits0) in states.items():
            for b in sorted(costs[l]):
                w = w0 + weights[l][b]
                if w > budget:
                    continue
                cand = (c0 + costs[l][b], bits0 + (b,))
                cur = nxt.get(w)
                if cur is None or (cand[0], tuple(-x for x in cand[1])) < (cur[0], tuple(-x for x in cur[1])):
                    nxt[w] = cand
        states = nxt
        if not states:
            raise ValueError("budget infeasible")
    w, (c, bits) = min(states.items(), key=lambda kv: (kv[1][0], kv[0], tuple(-x for x in kv[1][1])))
    return {"bits": list(bits), "cost": c, "bytes": w, "n_states_final": len(states)}
