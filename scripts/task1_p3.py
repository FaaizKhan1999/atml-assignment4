#!/usr/bin/env python3
"""P3 hyperparameter selection on the C4 DEV manifest only (pre-registered P3-design in docs/decisions.md).

Grid M x lambda for S in {0.50, 0.75}; lambda=0 (uniform = P2) evaluated as a non-selectable reference.
Writes results/task1/p3_selection.json (full grid = record of unselected designs).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.seed import set_seed  # noqa: E402
from atml4.task1 import data, p3  # noqa: E402
from atml4.task1.calib import load_stats  # noqa: E402
from atml4.task1.model import Pristine, load_base_model  # noqa: E402
from atml4.task1.prune import xnorms_from_stats  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
set_seed(6304, deterministic=False)
dev = torch.device(a.device)
torch.cuda.set_device(dev)
t_all = time.perf_counter()
model, _ = load_base_model(a.device)
pr = Pristine(model)
stats, _ = load_stats(ROOT / "results/task1/baseline/calib_input_stats.npz")
xn = xnorms_from_stats(stats)
recs = data.load("development")
t = time.perf_counter()
fp16 = p3.dev_nll(model, recs, dev)
t_fp16 = time.perf_counter() - t
t = time.perf_counter()
D = p3.block_outlier_ratios(model, xn)
t_D = time.perf_counter() - t
grid = []
for S in (0.50, 0.75):
    for M, lam in [(None, 0.0)] + [(M, lam) for M in p3.GRID_M for lam in p3.GRID_LAMBDA]:
        t = time.perf_counter()
        pr.restore(verify=False)
        info = p3.apply_p3(model, xn, D[M or p3.GRID_M[0]], S, lam)
        r = p3.dev_nll(model, recs, dev)
        torch.cuda.synchronize(dev)
        grid.append({"S": S, "M": M, "lambda": lam, "selectable": lam > 0, "dev_mean_nll": r["mean_nll"],
                     "dev_ppl": r["ppl"], "dev_tokens": r["tokens"], "s_blocks": info["s_blocks"],
                     "per_block_zeros": info["per_block_zeros"], "seconds": time.perf_counter() - t})
        print(f"S={S} M={M} lam={lam}: dev ppl {r['ppl']:.4f}", flush=True)
pr.restore(verify=True)
selected = {}
for S in (0.50, 0.75):
    cands = [g for g in grid if g["S"] == S and g["selectable"]]
    best = min(cands, key=lambda g: (g["dev_ppl"], g["lambda"], g["M"]))
    ref = next(g for g in grid if g["S"] == S and not g["selectable"])
    selected[str(S)] = {"M": best["M"], "lambda": best["lambda"], "dev_ppl": best["dev_ppl"],
                        "dev_mean_nll": best["dev_mean_nll"], "p2_uniform_dev_ppl": ref["dev_ppl"],
                        "p2_uniform_dev_mean_nll": ref["dev_mean_nll"], "s_blocks": best["s_blocks"],
                        "D": D[best["M"]]}
    print(f"SELECTED S={S}: M={best['M']} lambda={best['lambda']} dev ppl {best['dev_ppl']:.4f} "
          f"(P2 uniform {ref['dev_ppl']:.4f})", flush=True)
payload = {"selected": selected, "grid": grid, "D": {str(M): v for M, v in D.items()}, "dev_fp16": fp16,
           "selection_rule": "lowest dev PPL per S; ties -> smaller lambda, then smaller M; lambda=0 not selectable",
           "dev_manifest": "c4_development_locked (32 x 512, targets 1..511)",
           "gpu_seconds": {"total": time.perf_counter() - t_all, "grid_evals": sum(g["seconds"] for g in grid),
                           "outlier_ratios": t_D, "dev_fp16": t_fp16}}
save_result(ROOT / "results/task1/p3_selection.json", payload,
            collect_meta(seed=6304, dtype=torch.float16, models=["base"],
                         extra={"dataset_selection": data.selection_record(["development", "calibration"])}))
