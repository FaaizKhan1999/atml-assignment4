#!/usr/bin/env python3
"""Compact Task 1 summary with paired bootstrap 95% CIs (CPU).

Per condition: N / mean NLL / PPL per suite, HellaSwag acc & acc_norm (2000 where available, always first 512),
paired deltas vs baseline, sparsity, packed bytes, output error. CIs: 2,000 resamples, seed 6304; PPL resamples
whole windows and recomputes pooled NLL before exp; HellaSwag resamples examples. Extra contrasts: P2-P1 @50%,
Q2-Q1 @INT4 and @INT3. Writes results/task1/summary_fixed.{csv,json}.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from atml4.common.io import load_jsonl_result, load_result, save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.stats import paired_bootstrap_diff, pooled_nll_bootstrap  # noqa: E402
from atml4.task1.conditions import FIXED_CONDITIONS  # noqa: E402

R = ROOT / "results/task1"
SUITES = ("wikitext2", "c4")
CONTRASTS = [("p2_wanda_s50", "p1_mag_s50"), ("q2_awq_int4", "q1_int4"), ("q2_awq_int3", "q1_int3")]


def windows(cond, suite):
    p = R / cond / f"ppl_{suite}.jsonl"
    if not p.exists():
        return None
    _, rows = load_jsonl_result(p)
    rows = sorted(rows, key=lambda r: r["window_id"])
    return rows


def hs(cond):
    p = R / cond / "hellaswag.jsonl"
    if not p.exists():
        return None
    _, rows = load_jsonl_result(p)
    return sorted(rows, key=lambda r: r["order"])


def ppl_contrast(a_rows, b_rows):
    if a_rows is None or b_rows is None:
        return None
    if [r["window_id"] for r in a_rows] != [r["window_id"] for r in b_rows] or \
            [r["scored_tokens"] for r in a_rows] != [r["scored_tokens"] for r in b_rows]:
        raise ValueError("windows not paired")
    tok = [r["scored_tokens"] for r in a_rows]
    return pooled_nll_bootstrap([r["sum_nll"] for r in a_rows], tok, [r["sum_nll"] for r in b_rows])


def hs_contrast(a, b, n, metric):
    if a is None or b is None or len(a) < n or len(b) < n:
        return None
    a, b = a[:n], b[:n]
    if [x["order"] for x in a] != [x["order"] for x in b]:
        raise ValueError("hellaswag examples not paired")
    return paired_bootstrap_diff([x[metric] for x in a], [x[metric] for x in b], unit="example")


def ci_str(c, scale=1.0, fmt="{:+.4f}"):
    if c is None:
        return ""
    return f"{fmt.format(c['point'] * scale)} [{fmt.format(c['ci_low'] * scale)}, {fmt.format(c['ci_high'] * scale)}]"


def main():
    conds = ["baseline"] + [c for c in FIXED_CONDITIONS if (R / c / "summary.json").exists()]
    base_w = {s: windows("baseline", s) for s in SUITES}
    base_h = hs("baseline")
    out, csv_rows = {}, []
    for c in conds:
        _, s = load_result(R / c / "summary.json")
        w = {suite: windows(c, suite) for suite in SUITES}
        h = hs(c)
        rec = {"condition": c, "family": s.get("family", "baseline"), "hellaswag_n": len(h) if h else 0}
        for suite in SUITES:
            ps = s.get(f"ppl_{suite}", {})
            rec[suite] = {k: ps.get(k) for k in ("N", "mean_nll", "ppl")}
            if c != "baseline":
                rec[suite]["vs_baseline"] = ppl_contrast(w[suite], base_w[suite])
        hsum = s.get("hellaswag", {})
        rec["hellaswag"] = {"all": hsum.get("all"), "first_512": hsum.get("first_512")}
        if c != "baseline":
            rec["hellaswag"]["vs_baseline"] = {
                f"{m}@{n}": hs_contrast(h, base_h, n, m) for n in (2000, 512) for m in ("acc", "acc_norm")}
        st = s.get("storage", {})
        q = st.get("quantized")
        rec["storage"] = {
            "mask_pruned_fraction": st.get("mask_pruned_fraction"),
            "actual_zero_fraction": st.get("actual_zero_fraction"),
            "nominal_bits": (q or {}).get("nominal_bits_per_eligible_weight", st.get("nominal_bits_per_eligible_weight")),
            "dense_fp16_total_bytes": st.get("dense", {}).get("total_dense_fp16_bytes"),
            "sparse16_total_bytes": st.get("sparse16_estimate", {}).get("total_bytes"),
            "two_four_total_bytes": st.get("two_four_estimate", {}).get("total_bytes"),
            "quant_total_bytes": (q or {}).get("total_bytes"),
            "quant_eligible_packed_bytes": (q or {}).get("eligible_packed_bytes"),
            "serialized_awq_scale_bytes": (q or {}).get("serialized_awq_scale_bytes"),
        }
        oe = s.get("output_error")
        rec["output_error_param_weighted"] = oe.get("param_weighted_mean_rel") if oe else None
        lm = s.get("latency_memory", {})
        rec["latency_ms_median"] = {k: lm[k]["median"] for k in ("prefill_512", "prefill_2048",
                                                                  "decode_64_after_512") if k in lm}
        rec["peak_alloc_bytes"] = {k: lm[k]["max_memory_allocated_bytes"] for k in
                                   ("prefill_512", "prefill_2048", "decode_64_after_512") if k in lm}
        out[c] = rec
        hall = rec["hellaswag"]["all"] or {}
        h512 = rec["hellaswag"]["first_512"] or {}
        vb = rec["hellaswag"].get("vs_baseline", {})
        st_ = rec["storage"]
        csv_rows.append({
            "condition": c, "hs_n": rec["hellaswag_n"],
            "wt2_N": rec["wikitext2"]["N"], "wt2_mean_nll": rec["wikitext2"]["mean_nll"], "wt2_ppl": rec["wikitext2"]["ppl"],
            "wt2_dppl_ci": ci_str((rec["wikitext2"].get("vs_baseline") or {}).get("ppl"), fmt="{:+.3f}"),
            "c4_N": rec["c4"]["N"], "c4_mean_nll": rec["c4"]["mean_nll"], "c4_ppl": rec["c4"]["ppl"],
            "c4_dppl_ci": ci_str((rec["c4"].get("vs_baseline") or {}).get("ppl"), fmt="{:+.3f}"),
            "hs2000_acc": hall.get("acc") if rec["hellaswag_n"] >= 2000 else "",
            "hs2000_acc_norm": hall.get("acc_norm") if rec["hellaswag_n"] >= 2000 else "",
            "hs2000_dacc_norm_pp_ci": ci_str(vb.get("acc_norm@2000"), 100, "{:+.2f}"),
            "hs512_acc": h512.get("acc"), "hs512_acc_norm": h512.get("acc_norm"),
            "hs512_dacc_norm_pp_ci": ci_str(vb.get("acc_norm@512"), 100, "{:+.2f}"),
            "mask_pruned_frac": st_["mask_pruned_fraction"], "zero_frac": st_["actual_zero_fraction"],
            "nominal_bits": st_["nominal_bits"],
            "packed_total_bytes": st_["quant_total_bytes"] or st_["sparse16_total_bytes"],
            "packed_kind": "quant packed+meta+scales+other" if st_["quant_total_bytes"] else "sparse16 estimate",
            "two_four_total_bytes": st_["two_four_total_bytes"] or "",
            "outerr_pw_mean": rec["output_error_param_weighted"],
            "prefill512_ms": rec["latency_ms_median"].get("prefill_512"),
            "prefill2048_ms": rec["latency_ms_median"].get("prefill_2048"),
            "decode64_ms": rec["latency_ms_median"].get("decode_64_after_512"),
        })
    contrasts = {}
    for a, b in CONTRASTS:
        if a in out and b in out:
            ha, hb = hs(a), hs(b)
            contrasts[f"{a}-{b}"] = {
                **{suite: ppl_contrast(windows(a, suite), windows(b, suite)) for suite in SUITES},
                **{f"hs_{m}@{n}": hs_contrast(ha, hb, n, m) for n in (2000, 512) for m in ("acc", "acc_norm")}}
    payload = {"conditions": out, "contrasts": contrasts,
               "bootstrap": {"resamples": 2000, "seed": 6304, "ppl_unit": "window (pooled NLL then exp)",
                             "hellaswag_unit": "example"}}
    meta = collect_meta(seed=6304)
    save_result(R / "summary_fixed.json", payload, meta)
    tmp = R / "summary_fixed.csv"
    with open(tmp, "w", newline="") as f:
        f.write(f"# command: {meta['command']} | commit {meta['git']['commit']}\n")
        wr = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        wr.writeheader()
        wr.writerows(csv_rows)
    print(f"{len(conds)} conditions -> {R / 'summary_fixed.csv'}")
    for r in csv_rows:
        print(r["condition"], f"wt2 {r['wt2_ppl']:.3f} {r['wt2_dppl_ci']}", f"c4 {r['c4_ppl']:.3f} {r['c4_dppl_ci']}",
              f"hs512n {r['hs512_acc_norm']}", r["hs2000_dacc_norm_pp_ci"])
    for k, v in contrasts.items():
        print(k, {s: ci_str(v[s]["ppl"], fmt="{:+.3f}") for s in SUITES},
              "acc_norm@2000", ci_str(v["hs_acc_norm@2000"], 100, "{:+.2f}"))


if __name__ == "__main__":
    np.seterr(all="raise")
    main()
