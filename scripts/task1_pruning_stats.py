#!/usr/bin/env python3
"""Per-projection pruning diagnostics for RQ1 / P3: magnitude & Wanda score quantiles, OWL outlier ratios
(fraction of Wanda scores > M*mean, M in 3,5,7,10), Spearman(|W|, Wanda) on a seeded 200k-entry subsample,
and Jaccard overlap of magnitude vs Wanda pruned sets at s in {0.25, 0.5, 0.75}. Unchanged FP16 weights."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task1.calib import load_stats  # noqa: E402
from atml4.task1.model import eligible_projections, load_base_model  # noqa: E402
from atml4.task1.prune import score_diagnostics, xnorms_from_stats  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
t0 = time.time()
torch.cuda.set_device(a.device)
model, _ = load_base_model(a.device)
stats, rows = load_stats(ROOT / "results/task1/baseline/calib_input_stats.npz")
xn = xnorms_from_stats(stats)
out = []
with torch.no_grad():
    for i, p in enumerate(eligible_projections(model)):
        d = score_diagnostics(p.module.weight, xn[p.qualname], layer_index=i)
        out.append({"qualname": p.qualname, "layer": p.layer, "name": p.name, "shape": list(p.shape), **d})
save_result(ROOT / "results/task1/pruning_stats.json", {"per_projection": out, "seconds": time.time() - t0},
            collect_meta(seed=6304, dtype=torch.float16, models=["base"]))
print(f"pruning stats: {len(out)} projections in {time.time() - t0:.1f}s", flush=True)
