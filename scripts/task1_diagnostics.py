#!/usr/bin/env python3
"""Task 1 Step 9 diagnostics from saved files (CPU). Writes results/task1/diagnostics.json and qualitative.json.

* per-block D_l vs s_l vs block calibration output error (param-weighted over the block's 7 projections; P2 vs P3)
* Q3 probe dNLL vs chosen bits
* dev improvement vs test improvement (P3 vs P2, Q3 vs Q2); predicted vs realised dev dNLL (Q3, Q2 uniform)
* selection effort: GPU-seconds of grid / probes / search vs the compression itself
* qualitative HellaSwag cases by a fixed rule (first-512 IDs, manifest order)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy.stats import spearmanr  # noqa: E402

from atml4.common.io import load_jsonl_result, load_result, save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task1 import data, hellaswag  # noqa: E402

R = ROOT / "results/task1"
L = 28


def summ(c):
    return load_result(R / c / "summary.json")[1]


def block_err_paramweighted(s, numel):
    acc = [0.0] * L
    tot = [0] * L
    for x in s["output_error"]["per_projection"]:
        li = int(x["qualname"].split(".")[2])
        acc[li] += x["rel"] * numel[x["qualname"]]
        tot[li] += numel[x["qualname"]]
    return [a / t for a, t in zip(acc, tot)]


def sp(a, b):
    r = spearmanr(a, b)
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(a)}


def main():
    out = {}
    base = summ("baseline")
    numel = {}
    for m in base["per_matrix_nnz"]:
        numel[m["qualname"]] = m["numel"]

    # ---- P3 vs P2 per block
    sel = load_result(R / "p3_selection.json")[1]
    p3d = {}
    for S, p2c, p3c in (("0.5", "p2_wanda_s50", "p3_owl_s50"), ("0.75", "p2_wanda_s75", "p3_owl_s75")):
        s3 = summ(p3c)
        ci = s3["condition_info"]
        D = ci["D"]
        s_l = ci["s_blocks"]
        e2 = block_err_paramweighted(summ(p2c), numel)
        e3 = block_err_paramweighted(s3, numel)
        p3d[S] = {"M": ci["selected_M"], "lambda": ci["selected_lambda"], "D_l": D, "s_l": s_l,
                  "per_block_zeros": ci["per_block_zeros"], "block_outerr_P2": e2, "block_outerr_P3": e3,
                  "spearman_D_vs_s": sp(D, s_l), "spearman_D_vs_P2_err": sp(D, e2),
                  "spearman_s_vs_P3_err": sp(s_l, e3),
                  "spearman_s_minus_S_vs_err_change": sp([x - float(S) for x in s_l], [b - a for a, b in zip(e2, e3)])}
    out["p3_blocks"] = p3d

    # ---- Q3 probes vs bits
    qa = load_result(R / "q3_allocation.json")[1]
    dn = qa["probe_dnll"]
    q3d = {"probe_dnll": dn, "bits_options": qa["bits_options"], "allocations": {}}
    i4 = qa["bits_options"].index(4)
    for name, a in qa["allocations"].items():
        q3d["allocations"][name] = {"bits": a["bits"], "bit_counts": a["bit_counts"],
                                    "spearman_probe_dnll_at_4bit_vs_bits": sp([r[i4] for r in dn], a["bits"]),
                                    "predicted_dev_dnll": a["predicted_dev_dnll"],
                                    "realised_dev_dnll": a["realised_dev_dnll"], "total_bytes": a["total_bytes"],
                                    "budget_bytes": a["budget_bytes"]}
    q3d["q2_uniform_dev"] = qa["q2_uniform_dev"]
    q3d["alpha_rederivation_match"] = qa["alpha_rederivation_match"]
    out["q3"] = q3d

    # ---- Q3-proj (second design): sensitivity vs bits, per-type counts, predicted vs realised, primary
    if (R / "q3proj_allocation.json").exists():
        qp = load_result(R / "q3proj_allocation.json")[1]
        sens = qp["sensitivity_dnll"]
        j4 = qp["bits_options"].index(4)
        qpd = {"units": qp["units"], "bits_options": qp["bits_options"], "sensitivity_dnll": sens, "allocations": {}}
        for name, a in qp["allocations"].items():
            bits = [x["bits"] for x in a["bits_map"]]
            qpd["allocations"][name] = {
                "bits": bits, "counts_proj_type_x_bits": a["counts_proj_type_x_bits"],
                "per_block_param_weighted_mean_bits": [b["param_weighted_mean_bits"] for b in a["per_block"]],
                "spearman_probe_dnll_at_4bit_vs_bits": sp([r[j4] for r in sens], bits),
                "predicted_dev_dnll": a["predicted_dev_dnll"], "realised_dev_dnll": a["realised_dev_dnll"],
                "total_bytes": a["total_bytes"], "budget_bytes": a["budget_bytes"], "slack_bytes": a["slack_bytes"],
                "block_level_remeasured": a["block_level_remeasured"], "primary": a["primary"]}
        out["q3proj"] = qpd

    # ---- dev vs test carry-over
    def test_nll(c):
        s = summ(c)
        return {k: s[f"ppl_{k}"]["mean_nll"] for k in ("wikitext2", "c4")} | \
               {f"{k}_ppl": s[f"ppl_{k}"]["ppl"] for k in ("wikitext2", "c4")}

    carry = []
    for S, p2c, p3c in (("0.5", "p2_wanda_s50", "p3_owl_s50"), ("0.75", "p2_wanda_s75", "p3_owl_s75")):
        d = sel["selected"][S]
        t2, t3 = test_nll(p2c), test_nll(p3c)
        carry.append({"pair": f"{p3c} vs {p2c}", "dev_dnll": d["dev_mean_nll"] - d["p2_uniform_dev_mean_nll"],
                      "dev_ppl": [d["p2_uniform_dev_ppl"], d["dev_ppl"]],
                      "test_dnll_wikitext2": t3["wikitext2"] - t2["wikitext2"], "test_dnll_c4": t3["c4"] - t2["c4"]})
    for name, q2c, q3c, b in (("B4", "q2_awq_int4", "q3_mixed_b4", "int4"), ("B3", "q2_awq_int3", "q3_mixed_b3", "int3")):
        t2, t3 = test_nll(q2c), test_nll(q3c)
        a = qa["allocations"][name]
        carry.append({"pair": f"{q3c} vs {q2c}", "dev_dnll": a["realised_dev_dnll"] - qa["q2_uniform_dev"][b]["dev_dnll"],
                      "test_dnll_wikitext2": t3["wikitext2"] - t2["wikitext2"], "test_dnll_c4": t3["c4"] - t2["c4"]})
    if (R / "q3proj_allocation.json").exists() and (R / "q3proj_b4/summary.json").exists():
        qp = load_result(R / "q3proj_allocation.json")[1]["allocations"]
        for name, q2c, q3c, b in (("B4", "q2_awq_int4", "q3proj_b4", "int4"), ("B3", "q2_awq_int3", "q3proj_b3", "int3")):
            t2, t3 = test_nll(q2c), test_nll(q3c)
            carry.append({"pair": f"{q3c} vs {q2c}",
                          "dev_dnll": qp[name]["realised_dev_nll"] - qp[name]["block_level_remeasured"]["dev_nll"],
                          "dev_note": "vs block-level (= Q2 uniform) re-measured in the same process",
                          "test_dnll_wikitext2": t3["wikitext2"] - t2["wikitext2"], "test_dnll_c4": t3["c4"] - t2["c4"]})
    out["dev_vs_test"] = {"pairs": carry,
                          "same_sign_wikitext2": [math.copysign(1, x["dev_dnll"]) == math.copysign(1, x["test_dnll_wikitext2"])
                                                  for x in carry],
                          "note": "few method pairs only; a cross-pair correlation is not meaningful"}
    grid = sel["grid"]
    out["p3_dev_grid_compact"] = {S: {f"M{g['M']}_l{g['lambda']}": round(g["dev_ppl"], 4) for g in grid
                                      if str(g["S"]) == S} for S in ("0.5", "0.75")}

    # ---- selection effort
    def cs(c):
        return summ(c)["condition_info"].get("compress_seconds", {})
    out["selection_effort_gpu_seconds"] = {
        "p3_dev_grid": sel["gpu_seconds"], "q3": qa["gpu_seconds"],
        "q2_awq_capture_and_search_3bits": {k: v for k, v in cs("q2_awq_int4").items() if "shared" in k},
        "compression_apply": {c: cs(c) for c in ("p2_wanda_s50", "p3_owl_s50", "q2_awq_int4", "q3_mixed_b4", "q3proj_b4")
                              if (R / c / "summary.json").exists()},
        "q3proj": load_result(R / "q3proj_allocation.json")[1]["gpu_seconds"]
        if (R / "q3proj_allocation.json").exists() else None,
        "calibration_stats_baseline_s": base.get("phase_seconds", {}).get("calib_stats")}

    save_result(R / "diagnostics.json", out, collect_meta(seed=6304))

    # ---- qualitative (fixed rule)
    cfgs = ["baseline", "p1_mag_s50", "p2_wanda_s50", "p3_owl_s50", "q2_awq_int4", "q3_mixed_b4", "q2_awq_int3"]
    H = {c: {r["order"]: r for r in load_jsonl_result(R / c / "hellaswag.jsonl")[1]} for c in cfgs}
    recs = data.load("hellaswag")[:512]
    first512 = [r["order"] for r in recs]
    rule_a = [o for o in first512 if H["baseline"][o]["acc_norm"] == 1 and H["p2_wanda_s50"][o]["acc_norm"] == 0][:3]
    rule_b = [o for o in first512 if H["p2_wanda_s50"][o]["acc_norm"] == 0 and H["p3_owl_s50"][o]["acc_norm"] == 1][:2]
    rule_c = [o for o in first512 if H["p2_wanda_s50"][o]["acc_norm"] == 1 and H["p3_owl_s50"][o]["acc_norm"] == 0][:1]
    cases = []
    for tag, orders in (("baseline_correct_P2_wrong", rule_a), ("P2_wrong_P3_correct", rule_b),
                        ("P2_correct_P3_wrong", rule_c)):
        for o in orders:
            d = hellaswag.process_doc(recs[o]["example"])
            cases.append({"rule": tag, "order": o, "id": H["baseline"][o]["id"], "context": d["query"],
                          "endings": d["choices"], "gold": d["gold"],
                          "per_char_scores": {c: [ll / n for ll, n in zip(H[c][o]["loglik"], H[c][o]["char_len"])]
                                              for c in cfgs},
                          "pred_acc_norm": {c: H[c][o]["pred_acc_norm"] for c in cfgs}})
    save_result(R / "qualitative.json", {"rule": "first-512 IDs in manifest order: first 3 with baseline acc_norm "
                                         "correct and P2@50 wrong; first 2 with P2@50 wrong and P3@50 correct; first 1 "
                                         "with P2@50 correct and P3@50 wrong", "configs": cfgs, "cases": cases},
                collect_meta(seed=6304))
    print("diagnostics + qualitative written;", len(cases), "cases")


if __name__ == "__main__":
    main()
