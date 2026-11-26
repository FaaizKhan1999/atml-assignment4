#!/usr/bin/env python3
"""Task 3 analysis (CPU) from saved results; every section is skipped if its inputs are missing.

* Step 2 baseline: per gamma acceptance metrics, target passes/token, speculative/target-only time and decode-tps
  ratios (paired bootstrap 95% CIs over prompts, reps averaged per prompt), c and v_gamma, model-projected S_hat with
  alpha = pooled accepted/examined, break-even alpha (or none), EOS/truncation counts; fixed-64 diagnostic separately.
* Bank (iii) vs (i) consistency: source length / EOS rate, and the original draft's TV / forward KL on bank (i) vs
  (iii) over positions 1-64 (bootstrap CI over prompts of the difference of per-prompt means, unpaired by sequence).
* Distillation: updates, skipped, scored tokens, time by component, peak memory, params, merged-vs-unmerged error,
  loss curves (every update), off-policy collection cost.
* Step 5: pooled metrics by bank x draft, by workload and by position bin, seed dispersion, aligned disagreements.
* Standalone accuracy by model x workload.
Writes results/task3/summary.json.
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from atml4.common.io import load_jsonl_result, load_result, save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.stats import bootstrap_ci, paired_bootstrap_diff  # noqa: E402

R = ROOT / "results/task3"
RUNS = ["off_s6304", "off_s6305", "on_s6304", "on_s6305"]


def s_hat(alpha, gamma, c, v):
    return (1 + sum(alpha ** j for j in range(1, gamma + 1))) / (gamma * c + v)


def break_even(gamma, c, v):
    need = gamma * c + v
    if 1 + gamma <= need:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 1 + sum(mid ** j for j in range(1, gamma + 1)) > need:
            hi = mid
        else:
            lo = mid
    return hi


def baseline():
    if not (R / "baseline.jsonl").exists():
        return None
    _, rows = load_jsonl_result(R / "baseline.jsonl")
    costs = load_result(R / "baseline_costs.json")[1]
    out = {"costs": costs}
    for tag in ("primary_eos_aware_64", "diagnostic_fixed64_ignore_eos"):
        for g in (2, 4):
            rs = [r for r in rows if r["tag"] == tag and r["gamma"] == g]
            if not rs:
                continue
            ex = sum(r["spec"]["examined"] for r in rs)
            ac = sum(r["spec"]["accepted"] for r in rs)
            pr = sum(r["spec"]["proposed"] for r in rs)
            ntok = sum(r["spec"]["n_tokens"] for r in rs)
            ids = sorted({r["id"] for r in rs})
            per = defaultdict(list)
            for r in rs:
                per[r["id"]].append(r)
            spec_t = [statistics.mean(x["spec"]["total_s"] for x in per[i]) for i in ids]
            ar_t = [statistics.mean(x["ar"]["total_s"] for x in per[i]) for i in ids]
            alpha = ac / ex if ex else None
            rec = {"n_runs": len(rs), "accepted_over_examined": alpha, "accepted_over_proposed": ac / pr if pr else None,
                   "mean_accepted_prefix": statistics.mean(x for r in rs for x in [r["spec"]["mean_accepted_prefix"]] if x is not None),
                   "target_passes_per_token": sum(r["spec"]["target_passes"] for r in rs) / ntok,
                   "draft_passes_per_token": sum(r["spec"]["draft_passes"] for r in rs) / ntok,
                   "spec_tokens": ntok, "ar_tokens": sum(r["ar"]["n_tokens"] for r in rs),
                   "eos_spec": sum(r["spec"]["stop"] == "eos" for r in rs), "cap_spec": sum(r["spec"]["stop"] == "cap" for r in rs),
                   "eos_ar": sum(r["ar"]["stop"] == "eos" for r in rs),
                   "time_ratio_spec_over_ar_total": sum(spec_t) / sum(ar_t),
                   "per_prompt_time_ratio": bootstrap_ci([s / a for s, a in zip(spec_t, ar_t)], unit="prompt"),
                   "time_diff_spec_minus_ar_s": paired_bootstrap_diff(spec_t, ar_t, unit="prompt"),
                   "decode_tps_ratio_mean": statistics.mean(r["decode_tps_ratio_spec_over_ar"] for r in rs),
                   "peak_alloc_spec_max": max(r["spec"]["max_memory_allocated_bytes"] for r in rs),
                   "peak_alloc_ar_max": max(r["ar"]["max_memory_allocated_bytes"] for r in rs),
                   "round_time_s": {k: sum(r["spec"]["timing"][k] for r in rs) for k in rs[0]["spec"]["timing"]},
                   "by_workload": {w: {"accepted_over_examined": sum(r["spec"]["accepted"] for r in rs if r["workload"] == w)
                                       / max(1, sum(r["spec"]["examined"] for r in rs if r["workload"] == w)),
                                       "time_ratio": sum(r["spec"]["total_s"] for r in rs if r["workload"] == w)
                                       / sum(r["ar"]["total_s"] for r in rs if r["workload"] == w)}
                                   for w in sorted({r["workload"] for r in rs})}}
            proj = {}
            for L, c in costs["isolated"].items():
                cc, v = c["c"], c["v_gamma"][str(g)]
                proj[L] = {"c": cc, "v_gamma": v, "S_hat_at_measured_alpha": s_hat(alpha, g, cc, v) if alpha else None,
                           "break_even_alpha": break_even(g, cc, v)}
            rec["cost_model"] = proj
            out[f"{tag}|gamma{g}"] = rec
    return out


def bank_consistency():
    if not (R / "align_rows.jsonl.gz").exists():
        return None
    _, rows = load_jsonl_result(R / "align_rows.jsonl.gz")
    res = {}
    for metric in ("tv", "fkl"):
        per = {}
        for bank in ("teacher", "exact_spec_original_gamma4"):
            d = defaultdict(list)
            for r in rows:
                if r["bank"] == bank and r["draft"] == "original" and r["position"] <= 64:
                    d[r["prompt_id"]].append(r[metric])
            per[bank] = {k: statistics.mean(v) for k, v in d.items()}
        common = sorted(set(per["teacher"]) & set(per["exact_spec_original_gamma4"]))
        a = [per["exact_spec_original_gamma4"][k] for k in common]
        b = [per["teacher"][k] for k in common]
        res[metric] = {"spec_minus_teacher": paired_bootstrap_diff(a, b, unit="prompt"),
                       "note": "paired by prompt (independent sequences); positions 1-64; original draft"}
    return res


def alignment():
    if not (R / "align_rows.jsonl.gz").exists():
        return None
    from atml4.task3.align import aggregate

    _, rows = load_jsonl_result(R / "align_rows.jsonl.gz")
    short = [r for r in rows if r["position"] <= 64]
    out = {"by_bank_draft_pos1_64": aggregate(short, ("bank", "draft")),
           "by_bank_draft_workload_pos1_64": aggregate(short, ("bank", "draft", "workload")),
           "by_bank_draft_bin": aggregate(rows, ("bank", "draft", "bin"))}
    disp = {}
    for k, v in out["by_bank_draft_pos1_64"].items():
        bank, draft = k.split("|")
        if draft in RUNS:
            disp.setdefault(f"{bank}|{draft[:-6]}", []).append(v["tv"])
    out["seed_dispersion_tv"] = {k: {"values": v, "mean": float(np.mean(v)), "half_range": float(np.ptp(v) / 2)}
                                 for k, v in disp.items()}
    orig = {(r["prompt_id"], r["position"]): r for r in rows if r["bank"] == "teacher" and r["draft"] == "original"}
    on = {(r["prompt_id"], r["position"]): r for r in rows if r["bank"] == "teacher" and r["draft"] == "on_s6304"}
    diffs = sorted(((orig[k]["tv"] - on[k]["tv"], k) for k in orig if k in on), reverse=True)
    out["aligned_disagreements_rule"] = "bank (i), on_s6304 vs original: 3 largest TV improvements, 3 largest degradations"
    out["aligned_disagreements"] = [{"prompt_id": k[0], "position": k[1], "tv_original": orig[k]["tv"],
                                     "tv_on_s6304": on[k]["tv"], "improvement": d} for d, k in diffs[:3] + diffs[-3:]]
    return out


def distillation():
    out = {}
    for run in RUNS:
        p = R / f"distill_{run}.json"
        if not p.exists():
            continue
        d = load_result(p)[1]
        out[run] = {k: d[k] for k in ("updates", "skipped_updates", "scored", "time", "train_wall_s",
                                      "params", "peak_memory", "merge_s")}
        out[run]["presentations"] = len(d["presentations"])  # the log's list overwrote the trainer's count key
        out[run]["merged_vs_unmerged_max_abs_logit"] = max(e["max_abs_logit_diff"] for e in d["merged_vs_unmerged"])
        out[run]["merged_vs_unmerged_max_kl"] = max(e["max_kl_unmerged_merged"] for e in d["merged_vs_unmerged"])
        out[run]["loss_curve"] = [(u["update"], u["loss_mean_kl"], u["cum_scored"]) for u in d["updates_log"]]
        out[run]["termination_counts"] = {t: sum(x["termination"] == t for x in d["presentations"])
                                          for t in ("eos", "cap", "quota")}
        out[run]["mean_response_len"] = statistics.mean(x["response_len"] for x in d["presentations"])
    if (R / "teacher_collection.jsonl.gz").exists():
        m, rows = load_jsonl_result(R / "teacher_collection.jsonl.gz")
        out["teacher_collection"] = {k: m.get(k) for k in ("teacher_generation_seconds", "n", "tokens")} | \
            {"extra": m.get("extra", {})}
    return out or None


def accuracy():
    out = {}
    for model in ["original"] + RUNS:
        p = R / f"accuracy_{model}.jsonl"
        if not p.exists():
            continue
        _, rows = load_jsonl_result(p)
        rec = {}
        for w in ("math", "familiar", "shifted"):
            rs = [r for r in rows if r["workload"] == w]
            rec[w] = {"n": len(rs), "accuracy": statistics.mean(r["correct"] for r in rs),
                      "truncated": sum(r["truncated"] for r in rs)}
            if w == "math":
                rec[w]["unextractable"] = sum(r["unextractable"] for r in rs)
            else:
                rec[w]["format_compliant"] = statistics.mean(r["compliant"] for r in rs)
        out[model] = rec
    return out or None


DRAFTS = ["original"] + RUNS
LEN_BINS = [(1, 16), (17, 32), (33, 63), (64, 64)]  # 64 = the EOS-aware cap (n_tokens == 64)


def step6():
    out = {}
    if not (R / "step6_g2.jsonl").exists() and not (R / "step6_g4.jsonl").exists():
        return None
    dist = {r: load_result(R / f"distill_{r}.json")[1] for r in RUNS if (R / f"distill_{r}.json").exists()}
    coll_s = None
    if (R / "teacher_collection.jsonl.gz").exists():
        coll_s = load_jsonl_result(R / "teacher_collection.jsonl.gz")[0]["extra"]["teacher_generation_seconds"]
    tv1 = {}
    if (R / "align_rows.jsonl.gz").exists():
        _, rows_a = load_jsonl_result(R / "align_rows.jsonl.gz")
        for d in DRAFTS:
            xs = [r["acc1"] for r in rows_a if r["bank"] == "teacher" and r["draft"] == d and r["position"] <= 64]
            tv1[d] = statistics.mean(xs) if xs else None
    for g in (2, 4):
        p = R / f"step6_g{g}.jsonl"
        if not p.exists():
            continue
        _, rows = load_jsonl_result(p)
        costs = load_result(R / f"step6_g{g}_costs.json")[1]["isolated"]
        per = {}
        for d in DRAFTS:
            rs = [r for r in rows if r["draft"] == d]
            if not rs:
                continue
            ex = sum(r["spec"]["examined"] for r in rs)
            ac = sum(r["spec"]["accepted"] for r in rs)
            pr = sum(r["spec"]["proposed"] for r in rs)
            nt = sum(r["spec"]["n_tokens"] for r in rs)
            alpha = ac / ex if ex else None
            ids = sorted({r["id"] for r in rs})
            byp = defaultdict(list)
            for r in rs:
                byp[r["id"]].append(r)
            ratio_p = [statistics.mean(x["latency_ratio_spec_over_ar"] for x in byp[i]) for i in ids]
            acc_p = [sum(x["spec"]["accepted"] for x in byp[i]) / max(1, sum(x["spec"]["examined"] for x in byp[i])) for i in ids]
            spec_t = sum(r["spec"]["total_s"] for r in rs)
            ar_t = sum(r["ar"]["total_s"] for r in rs)
            S_meas = ar_t / spec_t
            cm = {}
            for L, c in costs[d].items():
                cc, v = c["c"], c["v_gamma"][str(g)]
                cm[L] = {"c": cc, "v_gamma": v, "S_hat": s_hat(alpha, g, cc, v), "break_even_alpha": break_even(g, cc, v)}
            comp = {k: sum(r["spec"]["timing"][k] for r in rs) for k in rs[0]["spec"]["timing"]}
            rec = {"n_runs": len(rs), "accepted_over_examined": alpha, "accepted_over_proposed": ac / pr if pr else None,
                   "mean_accepted_prefix": statistics.mean(r["spec"]["mean_accepted_prefix"] for r in rs
                                                           if r["spec"]["mean_accepted_prefix"] is not None),
                   "target_passes_per_token": sum(r["spec"]["target_passes"] for r in rs) / nt,
                   "draft_passes_per_token": sum(r["spec"]["draft_passes"] for r in rs) / nt,
                   "eos": sum(r["spec"]["stop"] == "eos" for r in rs), "cap": sum(r["spec"]["stop"] == "cap" for r in rs),
                   "latency_ratio_per_prompt": bootstrap_ci(ratio_p, unit="prompt"),
                   "decode_tps_ratio_mean": statistics.mean(r["decode_tps_ratio"] for r in rs),
                   "S_measured_total_time": S_meas, "cost_model": cm,
                   "round_decomposition_s": comp,
                   "prefill_s": sum(r["spec"]["prefill_s"] for r in rs),
                   "host_overhead_s": sum(r["spec"]["host_overhead_s"] for r in rs),
                   "peak_alloc_GB": max(r["spec"]["max_memory_allocated_bytes"] for r in rs) / 1e9,
                   "expected_acceptance_1mTV_teacher_bank": tv1.get(d),
                   "_ratio_p": dict(zip(ids, ratio_p)), "_acc_p": dict(zip(ids, acc_p)),
                   "mean_request_saving_s": (ar_t - spec_t) / len(rs)}
            def seg(xs):
                """segment metrics over runs xs: acceptance, target passes / token, latency ratio (mean over runs,
                95% CI clustered by prompt), counts."""
                return {"n_runs": len(xs), "n_prompts": len({r["id"] for r in xs}),
                        "accepted_over_examined": sum(r["spec"]["accepted"] for r in xs)
                        / max(1, sum(r["spec"]["examined"] for r in xs)),
                        "target_passes_per_token": sum(r["spec"]["target_passes"] for r in xs)
                        / sum(r["spec"]["n_tokens"] for r in xs),
                        "latency_ratio_runs": bootstrap_ci([r["latency_ratio_spec_over_ar"] for r in xs],
                                                           groups=[r["id"] for r in xs], unit="prompt (clustered runs)"),
                        "eos_at_64": sum(r["spec"]["n_tokens"] == 64 and r["spec"]["stop"] != "cap" for r in xs)}

            wl = {}
            for w in ("math", "familiar", "shifted"):
                ws = [i for i in ids if byp[i][0]["workload"] == w]
                wl[w] = {"latency_ratio": bootstrap_ci([rec["_ratio_p"][i] for i in ws], unit="prompt"),
                         "accepted_over_examined": sum(x["spec"]["accepted"] for i in ws for x in byp[i])
                         / max(1, sum(x["spec"]["examined"] for i in ws for x in byp[i])),
                         "target_passes_per_token": sum(x["spec"]["target_passes"] for i in ws for x in byp[i])
                         / sum(x["spec"]["n_tokens"] for i in ws for x in byp[i]),
                         "n_prompts": len(ws), "n_runs": sum(len(byp[i]) for i in ws)}
            rec["by_workload"] = wl
            lb = {}
            for lo, hi in LEN_BINS:
                xs = [r for r in rs if lo <= r["spec"]["n_tokens"] <= hi]
                if xs:
                    lb[f"[{lo},{hi}]"] = seg(xs)
            rec["by_output_length"] = lb
            if d != "original":
                ds = dist[d]
                up = ds["train_wall_s"] + ds["merge_s"] + (coll_s if d.startswith("off") and coll_s else 0.0)
                rec["upfront_cost_s"] = {"train_wall": ds["train_wall_s"], "merge": ds["merge_s"],
                                         "teacher_collection": coll_s if d.startswith("off") else 0.0, "total": up}
                sv = rec["mean_request_saving_s"]
                rec["deployment_break_even_requests_vs_target_only"] = up / sv if sv > 0 else "no break-even (slower than target-only)"
            per[d] = rec
        for d, rec in per.items():
            if d == "original":
                continue
            o = per["original"]
            common = sorted(set(rec["_ratio_p"]) & set(o["_ratio_p"]))
            rec["vs_original"] = {
                "latency_ratio": paired_bootstrap_diff([rec["_ratio_p"][i] for i in common], [o["_ratio_p"][i] for i in common], unit="prompt"),
                "acceptance": paired_bootstrap_diff([rec["_acc_p"][i] for i in common], [o["_acc_p"][i] for i in common], unit="prompt")}
            sv = o["mean_request_saving_s"] - rec["mean_request_saving_s"]
            rec["deployment_break_even_requests_vs_original_draft"] = rec["upfront_cost_s"]["total"] / -sv if sv < 0 else \
                "no break-even (not faster than the original draft)"
        for rec in per.values():
            rec.pop("_ratio_p")
            rec.pop("_acc_p")
        out[f"gamma{g}"] = per
        dp = R / f"step6_g{g}_diag.jsonl"
        if dp.exists():
            _, dg = load_jsonl_result(dp)
            out[f"gamma{g}_diag"] = {d: {"latency_ratio_mean": statistics.mean(r["latency_ratio_spec_over_ar"] for r in dg if r["draft"] == d),
                                         "accepted_over_examined": sum(r["spec"]["accepted"] for r in dg if r["draft"] == d)
                                         / max(1, sum(r["spec"]["examined"] for r in dg if r["draft"] == d)),
                                         "spec_s_total": sum(r["spec"]["total_s"] for r in dg if r["draft"] == d)}
                                     for d in sorted({r["draft"] for r in dg})}
    if (R / "merge_check.json").exists():
        out["merge_check"] = load_result(R / "merge_check.json")[1]
    return out


def main():
    payload = {"baseline": baseline(), "bank_consistency": bank_consistency(), "distillation": distillation(),
               "alignment": alignment(), "accuracy": accuracy(), "step6": step6()}
    save_result(R / "summary.json", payload, collect_meta(seed=6304))
    for k, v in payload.items():
        print(k, "present" if v else "missing")


if __name__ == "__main__":
    main()
