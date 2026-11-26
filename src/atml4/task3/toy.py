"""Three-symbol Markov toy for the exact-speculative-sampling chi-square check (PDF Task 3 Step 1)."""
from __future__ import annotations

import itertools

import torch

from .speculative import ToyMarkov, speculative_generate

# all entries >= 0.1 -> every length-3 sequence has probability >= 1e-3 (expected count >= 100 at 100,000 draws)
P_TARGET = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.25, 0.25, 0.5]], dtype=torch.float64)
PI_TARGET = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
P_DRAFT = torch.tensor([[0.2, 0.4, 0.4], [0.5, 0.2, 0.3], [0.1, 0.6, 0.3]], dtype=torch.float64)
PI_DRAFT = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
BOS = 3
SEQS = list(itertools.product(range(3), repeat=3))


def exact_joint() -> dict[tuple, float]:
    return {s: float(PI_TARGET[s[0]] * P_TARGET[s[0], s[1]] * P_TARGET[s[1], s[2]]) for s in SEQS}


def run(n: int, gamma: int, seed: int, residual: str = "positive") -> dict:
    target, draft = ToyMarkov(P_TARGET, PI_TARGET), ToyMarkov(P_DRAFT, PI_DRAFT)
    gen = torch.Generator().manual_seed(seed)
    counts = {s: 0 for s in SEQS}
    agg = {"proposed": 0, "accepted": 0, "examined": 0, "bonus": 0, "residual": 0, "rounds": 0}
    for _ in range(n):
        r = speculative_generate(target, draft, [BOS], 3, gamma, gen, residual=residual)
        counts[tuple(r["tokens"])] += 1
        for k in agg:
            agg[k] += r["stats"][k]
    return {"counts": counts, "stats": agg}


def chi_square(counts: dict, n: int) -> dict:
    from atml4.common.stats import chisquare_gof

    ex = exact_joint()
    obs = [counts[s] for s in SEQS]
    return chisquare_gof(obs, [ex[s] for s in SEQS])
