#!/usr/bin/env python3
"""Task 2C analysis (CPU): the matched-INT4-budget table, breakdowns, paired bootstrap contrasts, aligned
generations. Reads results/task2/partC/quality_s*.jsonl (+ timing.json, packed_bench.json if present).
Writes results/task2/partC/summary.json and summary_table.csv."""
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atml4.common.io import load_jsonl_result, load_result, save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.stats import paired_bootstrap_diff  # noqa: E402
from atml4.task2.kv import prompts as PR  # noqa: E402

R = ROOT / "results/task2/partC"
METHODS = ("fp16", "sink_recent", "qaware", "int4_tok", "int4_chan", "hybrid")
CONTRASTS = [(m, "fp16") for m in METHODS[1:]] + [("qaware", "sink_recent"), ("int4_chan", "int4_tok"),
                                                   ("hybrid", "qaware"), ("hybrid", "int4_chan")]
ALIGNED = [(2048, "middle", 8), (4096, "start", 8), (4096, "middle", 2)]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def main():
    rows = []
    for f in sorted(R.glob("quality_s*.jsonl")):
        rows += load_jsonl_result(f)[1]
    order = {c["id"]: i for i, c in enumerate(PR.load_cases())}
    rows.sort(key=lambda r: order[r["id"]])
    assert len(rows) == len({r["id"] for r in rows}), "duplicate prompts"
    n = len(rows)
    ref_ok = [r["methods"]["fp16"]["correct"] for r in rows]
    table, csv_rows = {}, []
    for m in METHODS:
        x = [r["methods"][m] for r in rows]
        cond = [xi["correct"] for xi, ok in zip(x, ref_ok) if ok]
        rec = {"n": n, "accuracy": mean([xi["correct"] for xi in x]),
               "conditional_accuracy_on_fp16_correct": mean(cond), "n_fp16_correct": len(cond),
               "ref_answer_nll_mean": mean([xi["ref_nll_sum"] for xi in x]),
               "kl_first_token_mean": mean([xi["kl_first_token_vs_fp16"] for xi in x]),
               "kl_first_token_median": statistics.median([xi["kl_first_token_vs_fp16"] for xi in x]),
               "first_token_argmax_agree_fp16": mean([xi["first_token_argmax"] == r["methods"]["fp16"]["first_token_argmax"]
                                                      for xi, r in zip(x, rows)]),
               "evidence_retained_fp16_mean": mean([xi["evidence_retained_fp16"] for xi in x]),
               "packed_bytes_mean": mean([xi["packed_bytes"] for xi in x]),
               "fraction_of_M16": {"mean": mean([xi["fraction_of_M16"] for xi in x]),
                                   "min": min(xi["fraction_of_M16"] for xi in x),
                                   "max": max(xi["fraction_of_M16"] for xi in x)},
               "fraction_of_M4_max": max(xi["fraction_of_M4"] for xi in x),
               "shared_decode_fp16_bytes_mean": mean([xi["shared_decode_fp16_bytes"] for xi in x]),
               "total_live_bytes_end_mean": mean([xi["total_live_bytes_end"] for xi in x]),
               "no_answer": sum(xi["answer"] is None for xi in x),
               "all_finite": all(xi["all_finite"] for xi in x)}
        if x[0]["recon_error"]:
            rec["recon_rel_fro"] = {k: mean([xi["recon_error"][k]["mean"] for xi in x]) for k in ("K", "V")}
            rec["recon_rel_fro_per_layer"] = {k: [mean([xi["recon_error"][k]["per_layer"][l] for xi in x])
                                                  for l in range(len(x[0]["recon_error"][k]["per_layer"]))]
                                              for k in ("K", "V")}
        if m in ("sink_recent", "qaware"):
            rec["n_kept_range"] = [min(xi["breakdown"]["n_kept"] for xi in x), max(xi["breakdown"]["n_kept"] for xi in x)]
        if m == "hybrid":
            rec["fp16_selected_range"] = [min(xi["breakdown"]["f_selected"] for xi in x),
                                          max(xi["breakdown"]["f_selected"] for xi in x)]
            rec["residual_fp16_range"] = [min(xi["breakdown"]["residual_fp16"] for xi in x),
                                          max(xi["breakdown"]["residual_fp16"] for xi in x)]
        table[m] = rec
        csv_rows.append({"method": m, "accuracy": rec["accuracy"], "cond_accuracy": rec["conditional_accuracy_on_fp16_correct"],
                         "ref_nll": rec["ref_answer_nll_mean"], "kl_mean": rec["kl_first_token_mean"],
                         "evidence_retained": rec["evidence_retained_fp16_mean"], "packed_bytes": rec["packed_bytes_mean"],
                         "frac_M16": rec["fraction_of_M16"]["mean"]})
    breakdown = {}
    for fi, fname in enumerate(("context_length", "evidence_position", "distractors")):
        levels = sorted({r["cell"][fi] for r in rows}, key=str)
        breakdown[fname] = {str(lv): {m: {"n": sum(r["cell"][fi] == lv for r in rows),
                                         "accuracy": mean([r["methods"][m]["correct"] for r in rows if r["cell"][fi] == lv]),
                                         "evidence_retained": mean([r["methods"][m]["evidence_retained_fp16"]
                                                                    for r in rows if r["cell"][fi] == lv])}
                                     for m in METHODS} for lv in levels}
    cells = {}
    for r in rows:
        cells.setdefault(str(tuple(r["cell"])), []).append(r)
    breakdown["cells_descriptive"] = {k: {m: mean([r["methods"][m]["correct"] for r in v]) for m in METHODS}
                                      for k, v in cells.items()}
    contrasts = {}
    for a, b in CONTRASTS:
        contrasts[f"{a}-{b}"] = {
            "accuracy": paired_bootstrap_diff([r["methods"][a]["correct"] for r in rows],
                                              [r["methods"][b]["correct"] for r in rows], unit="prompt"),
            "ref_answer_nll": paired_bootstrap_diff([r["methods"][a]["ref_nll_sum"] for r in rows],
                                                    [r["methods"][b]["ref_nll_sum"] for r in rows], unit="prompt")}
    cases = {c["id"]: c for c in PR.load_cases()}
    aligned = []
    for cl in ALIGNED:
        r = next((r for r in rows if tuple(r["cell"]) == cl), None)
        if r is None:
            continue
        c = cases[r["id"]]
        aligned.append({"rule": f"first prompt ID of cell {cl}", "id": r["id"], "L": r["L"], "code": r["code"],
                        "evidence_text": c["evidence_text"],
                        "evidence_tokens": r.get("evidence_token_range"),
                        "outputs": {m: r["methods"][m]["text"] for m in METHODS},
                        "correct": {m: r["methods"][m]["correct"] for m in METHODS},
                        "evidence_retained": {m: r["methods"][m]["evidence_retained_fp16"] for m in METHODS},
                        "fp16_selection_rle": {m: r["methods"][m].get("fp16_selection_rle")
                                               for m in ("sink_recent", "qaware", "hybrid")}})
    payload = {"n_prompts": n, "fp16_reference_accuracy": mean(ref_ok), "table": table, "breakdown": breakdown,
               "contrasts": contrasts, "aligned_generations": aligned,
               "bootstrap": {"resamples": 2000, "seed": 6304, "unit": "prompt"}}
    for name in ("timing.json", "packed_bench.json"):
        if (R / name).exists():
            payload[name.split(".")[0]] = load_result(R / name)[1]
    meta = collect_meta(seed=6304)
    save_result(R / "summary.json", payload, meta)
    with open(R / "summary_table.csv", "w", newline="") as f:
        f.write(f"# command: {meta['command']} | commit {meta['git']['commit']}\n")
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    for r in csv_rows:
        print(r)
    for k, v in contrasts.items():
        c = v["accuracy"]
        print(f"{k:22s} dAcc {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]")


if __name__ == "__main__":
    main()
