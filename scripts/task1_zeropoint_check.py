#!/usr/bin/env python3
"""Task 1 safety check (CPU): Task 1's quantizer clamps the zero-point z = clamp(round(-min/s), 0, 2^b-1). The clamp
can only change z for a group with min > 0 or max < 0 (otherwise -min/s lies in [0, 2^b-1]). Count such groups
(group 128 along input columns, as quantize() forms them) in every matrix actually quantized:
  * Q1: the 196 eligible FP16 weights (bit-independent: the group sign pattern does not depend on b)
  * Q2 / Q3 (block) / Q3-proj: AWQ-scaled matrices W diag(s) exactly as awq.apply_quantized(_proj) builds them, for
    every bit width with saved scales (8/4/3 from q2_awq_search.npz, 2 from q3_awq_search.npz), including up_proj
    folded with down scales of every down bit width (and unfolded, for Q3-proj probes / allocations with FP16 down).
Writes results/task1/zeropoint_clamp_check.json."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta, model_revision  # noqa: E402
from atml4.task1 import awq  # noqa: E402

G = 128


def nonstraddling(W: torch.Tensor) -> tuple[int, int]:
    x = W.float().view(W.shape[0], -1, G)
    mn, mx = x.amin(-1), x.amax(-1)
    return int(((mn > 0) | (mx < 0)).sum()), mn.numel()


def main():
    mid, rev = model_revision("base")
    model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.float16).eval()
    q2 = awq.load_search(ROOT / "results/task1/q2_awq_search.npz")
    q3 = awq.load_search(ROOT / "results/task1/q3_awq_search.npz")
    res_by_bits = {8: q2, 4: q2, 3: q2, 2: q3}
    out = {"q1_unscaled": {"bad": 0, "groups": 0}, "awq_scaled": {}, "per_matrix_nonzero": []}
    for li, layer in enumerate(model.model.layers):
        mods = awq._mods(layer)
        for n, m in mods.items():
            b, g = nonstraddling(m.weight)
            out["q1_unscaled"]["bad"] += b
            out["q1_unscaled"]["groups"] += g
            if b:
                out["per_matrix_nonzero"].append({"kind": "q1", "layer": li, "proj": n, "bad": b})
        for bits, r in res_by_bits.items():
            S = {gname: torch.from_numpy(r["scales"][f"L{li}.{gname}.b{bits}"]).float() for gname, _ in awq.GROUPS}
            mats = {n: mods[n].weight.float() * S[awq.PROJ_GROUP[n]][None, :] for n in awq.PROJ_NAMES if n != "up_proj"}
            mats["up_proj(unfolded)"] = mods["up_proj"].weight.float() * S["gateup"][None, :]
            for bd, rd in res_by_bits.items():  # up folded with down scales of every down bit width
                sd = torch.from_numpy(rd["scales"][f"L{li}.down.b{bd}"]).float()
                mats[f"up_proj(folded_down_b{bd})"] = (mods["up_proj"].weight.float() / sd[:, None]) * S["gateup"][None, :]
            for n, W in mats.items():
                b, g = nonstraddling(W)
                k = f"b{bits}"
                rec = out["awq_scaled"].setdefault(k, {"bad": 0, "groups": 0})
                rec["bad"] += b
                rec["groups"] += g
                if b:
                    out["per_matrix_nonzero"].append({"kind": f"awq_b{bits}", "layer": li, "proj": n, "bad": b})
        print(li, out["q1_unscaled"]["bad"], {k: v["bad"] for k, v in out["awq_scaled"].items()}, flush=True)
    out["all_zero"] = out["q1_unscaled"]["bad"] == 0 and all(v["bad"] == 0 for v in out["awq_scaled"].values())
    save_result(ROOT / "results/task1/zeropoint_clamp_check.json", out, collect_meta(seed=6304, models=["base"]))
    print("ALL ZERO:", out["all_zero"])


if __name__ == "__main__":
    main()
