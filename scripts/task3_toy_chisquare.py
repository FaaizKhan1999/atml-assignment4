#!/usr/bin/env python3
"""Task 3 Step 1 toy check (CPU): 100,000 fixed-seed length-3 sequences from the exact speculative sampler at
gamma in {1,2,3} on a 3-symbol Markov target/draft, Pearson chi-square vs the exact joint law; negative control with
the residual replaced by p. Writes results/task3/toy_chisquare.json (exact vs empirical joint tables)."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task3 import toy  # noqa: E402

N = 100_000


def main():
    ex = toy.exact_joint()
    out = {"P_target": toy.P_TARGET.tolist(), "pi_target": toy.PI_TARGET.tolist(), "P_draft": toy.P_DRAFT.tolist(),
           "pi_draft": toy.PI_DRAFT.tolist(), "n": N, "runs": []}
    for residual in ("positive", "p"):
        for gamma in (1, 2, 3):
            t = time.time()
            r = toy.run(N, gamma, 6304 + gamma, residual=residual)
            cs = toy.chi_square(r["counts"], N)
            out["runs"].append({"sampler": "correct" if residual == "positive" else "negative_control_residual_p",
                                "gamma": gamma, "seed": 6304 + gamma, "chi_square": cs, "stats": r["stats"],
                                "seconds": time.time() - t,
                                "joint": [{"seq": list(s), "exact": ex[s], "empirical": r["counts"][s] / N,
                                           "count": r["counts"][s], "expected_count": ex[s] * N} for s in toy.SEQS]})
            print(residual, gamma, cs, flush=True)
    save_result(ROOT / "results/task3/toy_chisquare.json", out, collect_meta(seed=6304))


if __name__ == "__main__":
    main()
