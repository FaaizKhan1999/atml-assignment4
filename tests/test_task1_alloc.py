"""P3 OWL allocation and Q3 knapsack tests (CPU, synthetic)."""
import itertools
import random

import pytest

torch = pytest.importorskip("torch")

from atml4.task1 import p3, q3  # noqa: E402


@pytest.mark.parametrize("S,lam", [(0.5, 0.08), (0.75, 0.2), (0.5, 0.2), (0.75, 0.02)])
def test_owl_allocation_properties(S, lam):
    rng = random.Random(1)
    D = [rng.random() * 0.02 for _ in range(28)]
    w = [46792704] * 28
    s, c = p3.owl_allocation(D, S, lam, weights=w)
    assert all(p3.LO <= x <= p3.HI for x in s)
    assert sum(a * b for a, b in zip(s, w)) / sum(w) == pytest.approx(S, abs=1e-9)
    order = sorted(range(28), key=lambda i: D[i])
    assert all(s[order[i]] >= s[order[i + 1]] - 1e-12 for i in range(27))  # higher D -> lower (or equal) sparsity
    assert max(s) - min(s) > 0  # nonuniform
    # constant D -> uniform
    assert p3.owl_allocation([0.01] * 5, S, lam)[0] == pytest.approx([S] * 5)


def test_owl_clipping_active():
    D = list(range(10))
    s, _ = p3.owl_allocation(D, 0.84, 0.2)
    assert max(s) == pytest.approx(p3.HI) and min(s) >= p3.LO
    assert sum(s) / 10 == pytest.approx(0.84, abs=1e-9)


def test_row_counts_exact_total():
    shapes = [(l, d_out, d_in) for l in range(3) for d_out, d_in in [(8, 16), (4, 16), (16, 32)]]
    numel = sum(o * i for _, o, i in shapes)
    s_blocks = [0.31, 0.47, 0.72]
    target = int(0.5 * numel)
    ks, info = p3.row_counts(shapes, s_blocks, target)
    assert sum(int(k.sum()) for k in ks) == target
    assert sum(info["per_block_zeros"]) == target
    base = [round(s_blocks[l] * i) for l, o, i in shapes]
    for k, b in zip(ks, base):  # each row moved by at most 1 from its base count
        assert int((k - b).abs().max()) <= 1


def test_row_mask_k_matches_row_mask():
    from atml4.task1.prune import row_mask, row_mask_k

    S = torch.rand(6, 16, generator=torch.Generator().manual_seed(0))
    assert torch.equal(row_mask_k(S, torch.full((6,), 8)), row_mask(S, 0.5))
    m = row_mask_k(S, torch.tensor([0, 1, 2, 3, 15, 16]))
    assert m.sum(1).tolist() == [0, 1, 2, 3, 15, 16]


def _brute(costs, weights, budget):
    best = None
    for bits in itertools.product(*[sorted(c) for c in costs]):
        w = sum(weights[l][b] for l, b in enumerate(bits))
        if w > budget:
            continue
        key = (sum(costs[l][b] for l, b in enumerate(bits)), w, tuple(-x for x in bits))
        if best is None or key < best[0]:
            best = (key, list(bits), w)
    return best


@pytest.mark.parametrize("seed", range(12))
def test_knapsack_optimal_small(seed):
    rng = random.Random(seed)
    n = 5
    wb = {2: 3, 3: 4, 4: 5, 8: 9}
    costs = [{b: round(rng.uniform(0, 1) / b, 3) for b in q3.BITS} for _ in range(n)]
    budget = rng.randint(3 * n, 9 * n)
    sol = q3.knapsack(costs, [wb] * n, budget)
    ref = _brute(costs, [wb] * n, budget)
    assert sol["bits"] == ref[1] and sol["bytes"] == ref[2] <= budget
    assert sol["cost"] == pytest.approx(ref[0][0])


def test_knapsack_ties_and_infeasible():
    costs = [{2: 1.0, 4: 0.0}, {2: 1.0, 4: 0.0}]
    sol = q3.knapsack(costs, [{2: 1, 4: 2}] * 2, 3)  # only one block can get 4 bits; prefer lower index
    assert sol["bits"] == [4, 2]
    with pytest.raises(ValueError):
        q3.knapsack(costs, [{2: 5, 4: 6}] * 2, 3)


def test_block_bytes_match_q2_budgets():
    shapes = [(1536, 1536), (256, 1536), (256, 1536), (1536, 1536), (8960, 1536), (8960, 1536), (1536, 8960)]
    other, ser = 467_037_184, 28 * 9216
    assert 28 * q3.block_packed_bytes(shapes, 4) + other + ser == 1_147_982_848
    assert 28 * q3.block_packed_bytes(shapes, 3) + other + ser == 982_928_896
