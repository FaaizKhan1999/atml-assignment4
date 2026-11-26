#!/usr/bin/env python3
"""Dilated alignment diagnostic (CPU, saved query outputs only; recorded in docs/decisions.md).

For dilated K4/K16 mixed and dense K4/K16 mixed (controls), 3 seeds each: test accuracy grouped by the
query-to-VALUE offset residue (q - (s + 1)) mod 4 and the query-to-KEY offset residue (q - s) mod 4, split
near (distance < 128) / far (distance >= 128) and by finer distance bands. Positions come from the frozen test
records; predictions from results/task2/partB/queries/<run>.jsonl.gz.
Writes results/task2/partB/dilated_alignment.json.
"""
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task2 import mqar_data  # noqa: E402
from atml4.task2.mqar_model import DILATION, N_DILATED, WINDOW  # noqa: E402

Q = ROOT / "results/task2/partB/queries"
GROUPS = [("dilated", 4), ("dilated", 16), ("dense", 4), ("dense", 16)]
SEEDS = (6304, 6305, 6306)
BANDS = {"near_lt128": (0, 128), "far_ge128": (128, 10 ** 6), "[0,64)": (0, 64), "[64,128)": (64, 128)}


def coverage():
    local = set(range(WINDOW))
    dil = {DILATION * m for m in range(N_DILATED)}
    one = local | dil
    two = {a + b for a in one for b in one}
    out = {"local_head_offsets": f"0..{WINDOW - 1}", "dilated_head_offsets": f"0,{DILATION},...,{DILATION * (N_DILATED - 1)}",
           "per_residue": {}}
    for r in range(4):
        far = [o for o in range(64, 256) if o % 4 == r]
        out["per_residue"][str(r)] = {
            "one_hop_offsets_ge64_covered_fraction": sum(o in one for o in far) / len(far),
            "two_hop_offsets_ge64_covered_fraction": sum(o in two for o in far) / len(far),
            "one_hop_note": "a single attention hop reaches offset o >= 64 only through the dilated heads, i.e. only "
                            "if o % 4 == 0",
        }
    out["induction_path"] = ("layer 1: value token at s+1 reads its key at offset 1 (local heads); layer 2: query reads "
                             "position s+1 at the value offset q-(s+1): direct only if q-(s+1) <= 63 or "
                             "q-(s+1) % 4 == 0. Two-hop relays (local+dilated) cover every residue up to offset 315 in "
                             "the receptive-field sense, but the key match would then have to be carried by the relay")
    return out


def main():
    recs = {r["id"]: r for r in mqar_data.load_split("test")}
    out = {"coverage": coverage(), "runs": {}, "pooled": {}}
    for arch, K in GROUPS:
        acc_g = {}
        for seed in SEEDS:
            name = f"{arch}_K{K}_mixed_s{seed}"
            rows = []
            with gzip.open(Q / f"{name}.jsonl.gz", "rt") as f:
                for line in f:
                    x = json.loads(line)
                    r = recs[x["id"]]
                    qp, sp = r["query_positions"][x["q"]], r["source_positions"][x["q"]]
                    assert qp - sp == x["distance"]
                    rows.append((x["cell"], qp - (sp + 1), qp - sp, x["distance"], x["pred"] == x["gold"]))
            a = np.array([(v, k, d, c) for _, v, k, d, c in rows], dtype=np.int64)
            cellname = np.array([c for c, *_ in rows])
            res = {}
            for scope, sel in (("all_cells", np.ones(len(a), bool)), (f"trained_cell_K{K}_mixed", cellname == f"K{K}_mixed")):
                res[scope] = {}
                for band, (lo, hi) in BANDS.items():
                    m = sel & (a[:, 2] >= lo) & (a[:, 2] < hi)
                    for kind, col in (("value_offset_mod4", 0), ("key_offset_mod4", 1)):
                        res[scope].setdefault(kind, {})[band] = {
                            str(rr): {"n": int((m & (a[:, col] % 4 == rr)).sum()),
                                      "acc": float(a[m & (a[:, col] % 4 == rr), 3].mean())
                                      if (m & (a[:, col] % 4 == rr)).any() else None} for rr in range(4)}
                for kind, col in (("value_offset_mod4", 0), ("key_offset_mod4", 1)):
                    for band, (lo, hi) in BANDS.items():
                        for rr in range(4):
                            m = sel & (a[:, 2] >= lo) & (a[:, 2] < hi) & (a[:, col] % 4 == rr)
                            acc_g.setdefault((scope, kind, band, rr), [0, 0])
                            acc_g[(scope, kind, band, rr)][0] += int(a[m, 3].sum())
                            acc_g[(scope, kind, band, rr)][1] += int(m.sum())
            out["runs"][name] = res
        pooled = {}
        for (scope, kind, band, rr), (c, n) in acc_g.items():
            pooled.setdefault(scope, {}).setdefault(kind, {}).setdefault(band, {})[str(rr)] = \
                {"n": n, "acc": c / n if n else None}
        out["pooled"][f"{arch}_K{K}_mixed"] = pooled
    save_result(ROOT / "results/task2/partB/dilated_alignment.json", out, collect_meta(seed=6304))
    for g, p in out["pooled"].items():
        for kind in ("value_offset_mod4", "key_offset_mod4"):
            for band in ("near_lt128", "far_ge128"):
                d = p[f"trained_cell_K{g.split('_K')[1].split('_')[0]}_mixed"][kind][band]
                print(f"{g:18s} {kind:18s} {band:11s} " + " ".join(f"r{r}={d[r]['acc']:.3f}(n={d[r]['n']})"
                                                                     for r in "0123"))


if __name__ == "__main__":
    main()
