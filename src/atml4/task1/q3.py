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


def proj_milp(costs: list[dict[int, float]], weights: list[dict[int, int]], vec_groups: list[dict],
              budget: int, fixed_bytes: int = 0, cost_scale: float = 1e6) -> dict:
    """Exact multiple-choice knapsack with shared scale-vector costs, via scipy.optimize.milp (HiGHS).

    costs[u][b], weights[u][b]: per-unit objective and packed bytes. vec_groups: list of
    {"units": [u,...], "vec_of_bits": {b: vector_id}, "vec_bytes": {vector_id: bytes}} -- a vector's bytes are
    paid once if ANY unit of the group uses a bit width mapping to it. Constraint: fixed_bytes + sum weights +
    sum vector bytes <= budget. Ties: second MILP minimising bytes s.t. objective <= optimum + 1e-6 (scaled units, micro-nats
    ). Returns bits per unit, objective (unscaled), integer bytes (recomputed exactly)."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    U = len(costs)
    xi = []  # (u, b)
    for u in range(U):
        for b in sorted(costs[u]):
            xi.append((u, b))
    yi = []  # (group index, vector id)
    for gi, g in enumerate(vec_groups):
        for v in sorted(set(g["vec_of_bits"].values())):
            yi.append((gi, v))
    nx, ny = len(xi), len(yi)
    n = nx + ny
    c_obj = np.zeros(n)
    w = np.zeros(n)
    for k, (u, b) in enumerate(xi):
        c_obj[k] = costs[u][b] * cost_scale
        w[k] = weights[u][b]
    ypos = {key: nx + k for k, key in enumerate(yi)}
    for (gi, v), k in ypos.items():
        w[k] = vec_groups[gi]["vec_bytes"][v]
    rows, lo, hi = [], [], []
    for u in range(U):  # one-hot
        r = np.zeros(n)
        for k, (uu, _) in enumerate(xi):
            if uu == u:
                r[k] = 1
        rows.append(r), lo.append(1), hi.append(1)
    xpos = {key: k for k, key in enumerate(xi)}
    for gi, g in enumerate(vec_groups):  # x[u,b] <= y[g, v(b)]
        for u in g["units"]:
            for b in costs[u]:
                r = np.zeros(n)
                r[xpos[(u, b)]] = 1
                r[ypos[(gi, g["vec_of_bits"][b])]] = -1
                rows.append(r), lo.append(-np.inf), hi.append(0)
    A = np.array(rows)
    cons = [LinearConstraint(A, lo, hi), LinearConstraint(w[None, :], -np.inf, budget - fixed_bytes)]
    opts = {"mip_rel_gap": 0.0, "presolve": True}
    integ = np.ones(n)
    bnd = Bounds(np.zeros(n), np.ones(n))
    r1 = milp(c_obj, constraints=cons, integrality=integ, bounds=bnd, options=opts)
    if r1.status != 0:
        raise ValueError(f"MILP failed: {r1.message}")
    opt = float(c_obj @ np.round(r1.x))
    cons2 = cons + [LinearConstraint(c_obj[None, :], -np.inf, opt + 1e-6)]
    r2 = milp(w, constraints=cons2, integrality=integ, bounds=bnd, options=opts)
    x = np.round(r2.x if r2.status == 0 else r1.x)
    bits = [None] * U
    for k, (u, b) in enumerate(xi):
        if x[k] > 0.5:
            bits[u] = b
    assert all(b is not None for b in bits)
    used = set()
    for gi, g in enumerate(vec_groups):
        for u in g["units"]:
            used.add((gi, g["vec_of_bits"][bits[u]]))
    total = fixed_bytes + sum(weights[u][bits[u]] for u in range(U)) + \
        sum(vec_groups[gi]["vec_bytes"][v] for gi, v in used)
    if total > budget:
        raise AssertionError(f"MILP solution exceeds budget: {total} > {budget}")
    return {"bits": bits, "cost": sum(costs[u][bits[u]] for u in range(U)), "bytes": int(total),
            "tie_break_status": int(r2.status), "status_msg": r1.message}


def brute_force(costs, weights, vec_groups, budget, fixed_bytes=0):
    """Reference exhaustive solver for tests (lexicographic: cost, then bytes)."""
    import itertools

    best = None
    for bits in itertools.product(*[sorted(c) for c in costs]):
        used = {(gi, g["vec_of_bits"][bits[u]]) for gi, g in enumerate(vec_groups) for u in g["units"]}
        tot = fixed_bytes + sum(weights[u][b] for u, b in enumerate(bits)) + \
            sum(vec_groups[gi]["vec_bytes"][v] for gi, v in used)
        if tot > budget:
            continue
        key = (round(sum(costs[u][b] for u, b in enumerate(bits)), 12), tot)
        if best is None or key < best[0]:
            best = (key, list(bits))
    return best
