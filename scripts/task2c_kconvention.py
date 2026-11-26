#!/usr/bin/env python3
"""CPU diagnostic: INT4 K quantization error on the REAL K/V saved by atml4-t2c-kdiag (kaggle_runs/t2c-kdiag/
output/kv_dumps, not committed) under three asymmetric conventions with FP16 scale + FP16 offset metadata:
  min_offset      : offset = FP16 round-down of min, scale rounded up, codes against stored values (current code)
  zp_unclamped    : Task 1 zero-point form, z = round(-min/s) NOT clamped (stored as an FP16 integer)
  zp_clamped_old  : Task 1 form with z clamped to [0, 15] (the original bug)
Writes results/task2/partC/k_quant_convention_compare.json."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task2.kv import kvquant as KQ  # noqa: E402

DUMPS = ROOT / "kaggle_runs/t2c-kdiag/output/kv_dumps"


def zp(Gx, clamp, bits=4):
    x = Gx.float()
    qmax = 2 ** bits - 1
    mn, mx = x.amin(-1), x.amax(-1)
    s = ((mx - mn) / qmax).clamp(min=2.0 ** -24).half().float()
    z = torch.round(-mn / s)
    if clamp:
        z = z.clamp(0, qmax)
    q = torch.clamp(torch.round(x / s[..., None]) + z[..., None], 0, qmax)
    return (q - z[..., None]) * s[..., None], z


FORMS = {"min_offset": lambda G: (KQ.quantize_groups(G, 4)[3], None),
         "zp_unclamped": lambda G: zp(G, False), "zp_clamped_old": lambda G: zp(G, True)}


def main():
    out = {}
    for f in sorted(DUMPS.glob("kv_*.pt")):
        d = torch.load(f)
        nq = (d["L"] // 64) * 64
        rec = {}
        for tensor in ("K_post", "V"):
            for lay in (("channel", "token") if tensor == "K_post" else ("token",)):
                for name, fn in FORMS.items():
                    per, num, den, zmax, zout = [], 0.0, 0.0, 0.0, []
                    for X in d[tensor]:
                        X = X[0, :, :nq]
                        y, z = fn(KQ._groups(X, lay))
                        xh = KQ._ungroup(y, lay, X.shape).half().float()
                        e = xh - X.float()
                        per.append(float(e.norm() / X.float().norm()))
                        num += float(e.pow(2).sum())
                        den += float(X.float().pow(2).sum())
                        if z is not None:
                            zmax = max(zmax, float(z.abs().max()))
                            zout.append(float(((z < 0) | (z > 15)).float().mean()))
                    rec[f"{tensor}/{lay}/{name}"] = {"pooled_rel": (num / den) ** 0.5, "mean_over_layers": sum(per) / len(per),
                                                     "per_layer": per, "max_abs_zero_point": zmax or None,
                                                     "frac_zero_point_outside_0_15_mean": sum(zout) / len(zout) if zout else None}
        out[d["id"]] = rec
        print(d["id"], {k: round(v["mean_over_layers"], 4) for k, v in rec.items()})
    save_result(ROOT / "results/task2/partC/k_quant_convention_compare.json", out,
                collect_meta(seed=6304, extra={"source": "kaggle_runs/t2c-kdiag/output/kv_dumps (atml4-t2c-kdiag @ 6621e46)"}))


if __name__ == "__main__":
    main()
