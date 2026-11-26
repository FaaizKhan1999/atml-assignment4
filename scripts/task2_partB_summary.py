#!/usr/bin/env python3
"""Aggregate Part B runs (CPU): per group mean +/- sd over seeds for every test cell, distance bins and dev curves.
Includes the B3 curriculum group (window_K16_curriculum) next to its comparator (window_K16_mixed).
Writes results/task2/partB/summary.json."""
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atml4.common.io import load_result, save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402

R = ROOT / "results/task2/partB/runs"


def ms(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": statistics.mean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0, "values": xs} if xs else None


def main():
    groups = {}
    for f in sorted(R.glob("*_s630?.json")):
        g = re.sub(r"_s630\d$", "", f.stem)
        groups.setdefault(g, []).append(load_result(f)[1])
    out = {}
    for g, runs in groups.items():
        cells = sorted(runs[0]["test"])
        rec = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs], "params": sorted({r["params"] for r in runs}),
               "test": {c: ms([r["test"][c]["acc"] for r in runs]) for c in cells},
               "dist_bins": {c: {b: ms([r["test"][c]["dist_bins"][b]["acc"] for r in runs])
                                 for b in runs[0]["test"][c]["dist_bins"]} for c in cells},
               "dev_curve_trained_cell": {str(p["update"]): ms([r["dev_curve"][i]["dev_trained_cell_acc"] for r in runs])
                                          for i, p in enumerate(runs[0]["dev_curve"])},
               "wall_seconds": ms([r.get("wall_seconds") for r in runs]),
               "mean_step_ms": ms([r.get("mean_step_ms") for r in runs]),
               "nonfinite_losses": sum(r.get("nonfinite_losses", 0) for r in runs)}
        if "curriculum" in runs[0]:
            rec["curriculum_mixed_examples_total"] = [r["curriculum"]["mixed_examples_total"] for r in runs]
        out[g] = rec
    save_result(ROOT / "results/task2/partB/summary.json", {"groups": out}, collect_meta(seed=6304))
    for g in ("window_K16_mixed", "window_K16_near", "window_K16_curriculum"):
        if g in out:
            t = out[g]["test"]
            print(g, {c: round(t[c]["mean"], 4) for c in ("K16_near", "K16_far", "K16_mixed")})


if __name__ == "__main__":
    main()
