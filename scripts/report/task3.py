#!/usr/bin/env python3
"""Task 3 report figures and table fragments (reads only results/task3/*). Outputs -> report/figs/task3/."""
from __future__ import annotations

import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import rstyle as R  # noqa: E402

S = "scripts/report/task3.py"
SUM = "results/task3/summary.json"
P = R.load(SUM)
RUNS = ["off_s6304", "off_s6305", "on_s6304", "on_s6305"]
DRAFTS = ["original"] + RUNS
LAB = {"original": "original", "off_s6304": "off s6304", "off_s6305": "off s6305", "on_s6304": "on s6304",
       "on_s6305": "on s6305"}
COL = {"original": R.OI["black"], "off_s6304": R.OI["blue"], "off_s6305": R.OI["sky"], "on_s6304": R.OI["verm"],
       "on_s6305": R.OI["orange"]}
BANKS = [("teacher", "(i) teacher"), ("original_draft", "(ii) orig. draft"), ("exact_spec_original_gamma4", "(iii) spec."),
         ("own", "own rollout")]
BINS = ["[1,16]", "[17,32]", "[33,64]", "[65,192]"]


def own_bank(d):
    return "original_draft" if d == "original" else f"own_{d}"


def align_rows():
    return R.load("results/task3/align_rows.jsonl.gz")


def fig6():
    w, h = 5.5, 1.85
    f = R.fig(w, h)
    a, b, c = f.subplots(1, 3, width_ratios=[1, 1.35, 1])
    for run in RUNS:
        ul = R.load(f"results/task3/distill_{run}.json")["updates_log"]
        x = np.cumsum([u["tokens"] for u in ul]) / 1000
        y = np.array([u["loss_mean_kl"] for u in ul])
        ls = "-" if run.endswith("6304") else "--"
        a.plot(x, y, color=COL[run], lw=0.3, alpha=0.35)
        k = 10
        ys = np.convolve(y, np.ones(k) / k, mode="valid")
        a.plot(x[k - 1:], ys, color=COL[run], lw=0.9, ls=ls, label=LAB[run])
    a.set_xlabel("teacher-scored tokens (k)")
    a.set_ylabel("KL(teacher||student) per token")
    a.legend(fontsize=5.2, ncol=2, loc="upper right", handlelength=1.4, columnspacing=0.5)
    rows = align_rows()
    acc = defaultdict(list)
    for r in rows:
        acc[(r["bank"], r["bin"], r["draft"])].append(r["tv"])
    xt, xl, pos = [], [], 0
    for bk, blab in BANKS:
        for bi, bn in enumerate(BINS):
            for k, d in enumerate(DRAFTS):
                bank = own_bank(d) if bk == "own" else bk
                v = acc.get((bank, bn, d))
                if v:
                    b.plot(pos + (k - 2) * 0.13, st.mean(v), marker="o", ms=2.0, color=COL[d], ls="none")
            xt.append(pos)
            xl.append(bn.strip("[]").replace(",", "-"))
            pos += 1
        b.text(pos - 2.5, 1.02, blab, transform=b.get_xaxis_transform(), ha="center", fontsize=5.3)
        if bk != "own":
            b.axvline(pos - 0.5, color=R.OI["grey"], lw=0.4)
    b.set_xticks(xt, xl, rotation=90, fontsize=4.8)
    b.set_xlim(-0.6, pos - 0.4)
    b.set_ylabel("TV(target, draft)")
    b.set_xlabel("response position bin, by prefix bank")
    for g, mk in ((2, "o"), (4, "s")):
        sp = P["step6"][f"gamma{g}"]
        for d in DRAFTS:
            v = sp[d]
            c.plot(v["expected_acceptance_1mTV_teacher_bank"], v["accepted_over_examined"], marker=mk, ms=3, color=COL[d],
                   ls="none", mfc=COL[d] if g == 2 else "none")
    lo, hi = 0.85, 0.89
    c.plot([lo, hi], [lo, hi], color=R.OI["grey"], lw=0.6, ls=":")
    c.set_xlim(lo, hi)
    c.set_ylim(lo, hi)
    c.set_xlabel("expected 1-TV (bank (i), pos 1-64)")
    c.set_ylabel("measured accepted/examined")
    hs = [Line2D([], [], color=COL[d], marker="o", ls="none", ms=3) for d in DRAFTS] + \
        [Line2D([], [], color="k", marker="o", ls="none", ms=3), Line2D([], [], color="k", marker="s", mfc="none", ls="none", ms=3)]
    c.legend(hs, [LAB[d] for d in DRAFTS] + ["$\\gamma$=2", "$\\gamma$=4"], fontsize=5.0, loc="lower right", ncol=2,
             handletextpad=0.1, columnspacing=0.4)
    for ax, l in zip((a, b, c), "abc"):
        R.panel(ax, l)
    R.record(R.save(f, 3, "fig6_distill_align.pdf", w, h), S,
             [SUM, "results/task3/align_rows.jsonl.gz"] + [f"results/task3/distill_{r}.json" for r in RUNS])


def fig7():
    w, h = 5.5, 1.85
    f = R.fig(w, h)
    a, b, c = f.subplots(1, 3, width_ratios=[1.3, 1, 1.1])
    comps = [("draft_s", "draft proposal", R.OI["blue"]), ("verify_s", "target verification", R.OI["orange"]),
             ("accept_s", "accept/residual", R.OI["green"]), ("cache_s", "cache rollback", R.OI["purple"]),
             ("prefill", "prefill", R.OI["grey"])]
    x, labels = 0, []
    for g in (2, 4):
        sp = P["step6"][f"gamma{g}"]
        for d in DRAFTS:
            v = sp[d]
            ntok = None
            rows = [r for r in R.load(f"results/task3/step6_g{g}.jsonl") if r["draft"] == d]
            ntok = sum(r["spec"]["n_tokens"] for r in rows)
            bottom = 0
            for k, lab, col in comps:
                val = (v["prefill_s"] if k == "prefill" else v["round_decomposition_s"][k]) / ntok * 1000
                a.bar(x, val, bottom=bottom, color=col, width=0.75, label=lab if x == 0 else None)
                bottom += val
            ar = sum(r["ar"]["total_s"] for r in rows) / sum(r["ar"]["n_tokens"] for r in rows) * 1000
            a.plot([x - 0.38, x + 0.38], [ar, ar], color="k", lw=0.8, label="target-only" if x == 0 else None)
            labels.append(f"{LAB[d]}")
            x += 1
        x += 0.6
    ticks = list(range(5)) + [5.6 + i for i in range(5)]
    a.set_xticks(ticks, labels, rotation=90, fontsize=4.8)
    a.text(2, 1.01, "$\\gamma$=2", transform=a.get_xaxis_transform(), ha="center", fontsize=5.5)
    a.text(7.6, 1.01, "$\\gamma$=4", transform=a.get_xaxis_transform(), ha="center", fontsize=5.5)
    a.set_ylabel("time per output token (ms)")
    a.legend(fontsize=4.8, loc="upper center", ncol=3, handlelength=0.8, columnspacing=0.4, bbox_to_anchor=(0.5, 0.99))
    a.set_ylim(0, a.get_ylim()[1] * 1.35)
    al = np.linspace(0.5, 1.0, 101)
    for g, ls in ((2, "-"), (4, "--")):
        cm = P["step6"][f"gamma{g}"]["original"]["cost_model"]["256"]
        sh = [(1 + sum(x_ ** j for j in range(1, g + 1))) / (g * cm["c"] + cm["v_gamma"]) for x_ in al]
        b.plot(al, sh, color=R.OI["grey"], ls=ls, lw=0.8, label=f"$\\hat S_{g}$ (c={cm['c']:.2f}, $v_{g}$={cm['v_gamma']:.2f})")
        for d in DRAFTS:
            v = P["step6"][f"gamma{g}"][d]
            b.plot(v["accepted_over_examined"], v["S_measured_total_time"], marker="o" if g == 2 else "s", ms=2.8,
                   color=COL[d], mfc=COL[d] if g == 2 else "none", ls="none")
    b.axhline(1, color="k", lw=0.5)
    b.set_xlabel("acceptance $\\alpha$")
    b.set_ylabel("speedup $S$")
    b.legend(fontsize=4.8, loc="upper left", handlelength=1.6)
    wl = ["math", "familiar", "shifted"]
    for g, mk in ((2, "o"), (4, "s")):
        for k, d in enumerate(DRAFTS):
            v = P["step6"][f"gamma{g}"][d]["by_workload"]
            for i, wname in enumerate(wl):
                ci_ = v[wname]["latency_ratio"]
                xx = i + (k - 2) * 0.08 + (0.42 if g == 4 else 0) - 0.21
                c.errorbar(xx, ci_["point"], yerr=[[ci_["point"] - ci_["ci_low"]], [ci_["ci_high"] - ci_["point"]]],
                           fmt=mk, ms=2.2, color=COL[d], mfc=COL[d] if g == 2 else "none", lw=0.5, capsize=0)
    c.axhline(1, color="k", lw=0.5)
    c.set_xticks(range(3), wl)
    c.set_ylabel("latency ratio spec / target-only")
    c.text(0.02, 0.97, "filled $\\gamma$=2 (left), open $\\gamma$=4 (right)", transform=c.transAxes, fontsize=4.8, va="top")
    for ax, l in zip((a, b, c), "abc"):
        R.panel(ax, l)
    R.record(R.save(f, 3, "fig7_spec_cost.pdf", w, h), S,
             [SUM, "results/task3/step6_g2.jsonl", "results/task3/step6_g4.jsonl"])


def tab3_sampler():
    t = R.load("results/task3/toy_chisquare.json")
    rows = []
    for g in (1, 2, 3):
        cor = next(r for r in t["runs"] if r["gamma"] == g and r["sampler"] == "correct")["chi_square"]
        neg = next(r for r in t["runs"] if r["gamma"] == g and r["sampler"] != "correct")["chi_square"]
        rows.append([str(g), f"{cor['statistic']:.1f}", f"{cor['p_value']:.3f}", f"{neg['statistic']:.0f}",
                     "$<10^{-16}$" if neg["p_value"] == 0 else f"{neg['p_value']:.1e}"])
    rows.append("MID")
    rows.append(["\\multicolumn{5}{@{}l@{}}{\\parbox{4.2in}{\\footnotesize Also passed (CPU tests): cache rollback with "
                 "history-dependent toy LMs (identical draft = 100\\% acceptance over many rounds), rejection rollback, "
                 "bonus path, EOS as accepted / bonus / residual token, budget truncation and one-token rounds, tiny-HF "
                 "full acceptance and post-run cache = fresh forward; real-model split prefill and cache-length checks "
                 "(T3-a).}}"])
    p = R.write_tab(3, "tab3_sampler.tex", "lrrrr", ["$\\gamma$", "$\\chi^2$ (exact)", "$p$",
                                                     "$\\chi^2$ (residual$\\to p$ control)", "$p$"], rows,
                    "% 100,000 length-3 sequences, 27 categories (26 dof), min expected count 500")
    R.record(p, S, ["results/task3/toy_chisquare.json"])


def tab3_spec():
    rows = []
    for g in (2, 4):
        for d in DRAFTS:
            v = P["step6"][f"gamma{g}"][d]
            rows.append([f"{LAB[d]}, $\\gamma$={g}", f"{v['accepted_over_examined']:.3f}", f"{v['accepted_over_proposed']:.3f}",
                         f"{v['mean_accepted_prefix']:.2f}", f"{v['target_passes_per_token']:.3f}",
                         R.ci(v["latency_ratio_per_prompt"], 1, "{:.2f}", signed=False), f"{v['S_measured_total_time']:.3f}",
                         f"{v['decode_tps_ratio_mean']:.2f}"])
        if g == 2:
            rows.append("MID")
    p = R.write_tab(3, "tab3_spec.tex", "lrrrrrrr",
                    ["Draft", "acc/exam.", "acc/prop.", "prefix", "passes/tok", "latency ratio [CI]", "$S$ (total)",
                     "tok/s ratio"], rows,
                    "% 120 prompts x 2 reps, EOS-aware 64-token cap; latency ratio = per-prompt mean of spec/adjacent "
                    "target-only; S = total target-only time / total speculative time; peak allocated 8.23 GB for every "
                    "condition (all five drafts resident in the benchmark process)")
    R.record(p, S, [SUM])


def tab3_costs():
    dist, acc = P["distillation"], P["accuracy"]
    coll = R.load("results/task3/teacher_collection.jsonl.gz")
    from atml4.common.io import load_jsonl_result
    coll_s = load_jsonl_result(R.ROOT / "results/task3/teacher_collection.jsonl.gz")[0]["extra"]["teacher_generation_seconds"]
    del coll
    rows = []
    for d in DRAFTS:
        a = acc[d]
        s4 = P["step6"]["gamma4"][d]
        if d == "original":
            cost = ["", "", "", "", "", ""]
            be = ""
        else:
            x = dist[d]
            t = x["time"]
            cost = [f"{coll_s:.0f}" if d.startswith("off") else "0", f"{t['student_rollout_s']:.0f}",
                    f"{t['teacher_scoring_s']:.0f}", f"{t['fwd_bwd_opt_s']:.0f}", f"{x['merge_s']:.1f}", str(x["updates"])]
            v = s4["deployment_break_even_requests_vs_target_only"]
            be = "none" if isinstance(v, str) else f"{v:.0f}"
        rows.append([LAB[d], *cost, f"{a['math']['accuracy']:.3f}", f"{a['familiar']['accuracy']:.3f}",
                     f"{a['shifted']['accuracy']:.3f}", be])
    p = R.write_tab(3, "tab3_costs.tex", "lrrrrrrrrrr",
                    ["Draft", "gen.", "roll.", "score", "f/b/opt", "merge", "upd.", "GSM8K", "fam.", "shift.", "BE"], rows,
                    "% costs in seconds: gen. = teacher-response collection (generated once, counted in both off-policy runs), roll. = "
                    "student rollouts, score = teacher scoring, f/b/opt = forward/backward/optimizer; BE (break-even) = "
                    "requests to amortise distillation + merge vs target-only (gamma=4): none, speculation is slower")
    R.record(p, S, [SUM, "results/task3/teacher_collection.jsonl.gz"])


# ---------------------------------------------------------------- appendix
def tabA_align():
    rows = align_rows()
    acc = defaultdict(list)
    for r in rows:
        if r["position"] <= 64:
            acc[(r["bank"], r["workload"], r["draft"])].append((r["tv"], r["fkl"]))
    out = []
    banks = [("teacher", "(i)"), ("original_draft", "(ii)"), ("exact_spec_original_gamma4", "(iii)")] + \
        [(f"own_{r}", f"own {LAB[r]}") for r in RUNS]
    for bk, bl in banks:
        for wname in ("math", "familiar", "shifted"):
            cells = []
            for d in DRAFTS:
                v = acc[(bk, wname, d)]
                cells.append(f"{st.mean(x[0] for x in v):.3f} / {st.mean(x[1] for x in v):.3f}")
            out.append([bl, wname] + cells)
        out.append("MID")
    out.pop()
    p = R.write_tab(3, "tabA_t3_align_domain.tex", "ll" + "r" * 5, ["Bank", "Workload"] + [LAB[d] for d in DRAFTS], out,
                    "% TV / forward KL, positions 1-64, token-weighted; LANDSCAPE recommended")
    R.record(p, S, ["results/task3/align_rows.jsonl.gz"])
    acc = defaultdict(list)
    for r in rows:
        acc[(r["bank"], r["bin"], r["draft"])].append(r)
    out = []
    for bk, bl in banks:
        for bn in BINS:
            cells = []
            for d in DRAFTS:
                v = acc[(bk, bn, d)]
                cells.append(f"{st.mean(x['tv'] for x in v):.3f}/{st.mean(x['greedy_agree'] for x in v):.2f}" if v else "")
            out.append([bl, bn, str(len(acc[(bk, bn, 'original')]))] + cells)
        out.append("MID")
    out.pop()
    p = R.write_tab(3, "tabA_t3_align_bins.tex", "llr" + "r" * 5, ["Bank", "Positions", "n"] + [LAB[d] for d in DRAFTS], out,
                    "% TV / greedy agreement by response-position bin incl. the long bank 65-192; LANDSCAPE recommended")
    R.record(p, S, ["results/task3/align_rows.jsonl.gz"])


def tabA_misc():
    d = P["alignment"]["aligned_disagreements"]
    rows = [[x["prompt_id"], str(x["position"]), f"{x['tv_original']:.3f}", f"{x['tv_on_s6304']:.3f}",
             R.num(x["improvement"], "{:.3f}", True)] for x in d]
    p = R.write_tab(3, "tabA_t3_disagree.tex", "lrrrr", ["Prompt", "pos.", "TV original", "TV on s6304", "improvement"], rows,
                    "% rule: bank (i), 3 largest TV improvements and 3 largest degradations, on-policy s6304 vs original")
    R.record(p, S, [SUM])
    b = P["baseline"]
    rows = []
    for k in ("primary_eos_aware_64|gamma2", "primary_eos_aware_64|gamma4", "diagnostic_fixed64_ignore_eos|gamma2",
              "diagnostic_fixed64_ignore_eos|gamma4"):
        v = b[k]
        rows.append([k.replace("_", " ").replace("|", ", "), f"{v['accepted_over_examined']:.3f}", f"{v['accepted_over_proposed']:.3f}",
                     f"{v['target_passes_per_token']:.3f}", f"{v['time_ratio_spec_over_ar_total']:.3f}",
                     R.ci(v["per_prompt_time_ratio"], 1, "{:.2f}", signed=False)])
    rows.append("MID")
    for L, c in b["costs"]["isolated"].items():
        rows.append([f"context {L}: c / $v_2$ / $v_4$", f"{c['c']:.3f}", f"{c['v_gamma']['2']:.3f}", f"{c['v_gamma']['4']:.3f}",
                     f"{c['target_1tok']['median_ms']:.1f} ms", f"{c['draft_1tok']['median_ms']:.1f} ms"])
    p = R.write_tab(3, "tabA_t3_baseline.tex", "lrrrrr",
                    ["Step 2 condition", "acc/exam.", "acc/prop.", "passes/tok", "time ratio", "per-prompt [CI]"], rows,
                    "% bottom block: c, v2, v4, target and draft single-step median at that context length")
    R.record(p, S, [SUM])
    rows = []
    for m, v in P["accuracy"].items():
        rows.append([LAB[m], f"{v['math']['accuracy']:.3f}", str(v["math"]["truncated"]), str(v["math"]["unextractable"]),
                     f"{v['familiar']['accuracy']:.3f}", str(v["familiar"]["truncated"]), f"{v['shifted']['accuracy']:.3f}",
                     str(v["shifted"]["truncated"])])
    p = R.write_tab(3, "tabA_t3_accuracy.tex", "lrrrrrrr",
                    ["Model", "GSM8K", "trunc.", "unextr.", "familiar", "trunc.", "shifted", "trunc."], rows,
                    "% 2 reps x 60 prompts per domain; caps 512 / 32; short-form accuracy = exact format compliance")
    R.record(p, S, [SUM])
    mc = P["step6"]["merge_check"]
    dg = P["step6"]["gamma4_diag"]
    rows = [[LAB[r], f"{mc[r]['mean_kl']:.1e}".replace("e-0", "e-"), f"{mc[r]['p99_kl']:.1e}".replace("e-0", "e-"),
             f"{mc[r]['max_kl']:.1e}".replace("e-0", "e-"), f"{mc[r]['argmax_agree']:.4f}"] for r in RUNS]
    rows.append("MID")
    rows.append(["unmerged off s6305 ($\\gamma$=4, fixed-64)", f"{dg['unmerged_off_s6305']['latency_ratio_mean']:.3f}",
                 "", f"merged {dg['off_s6305']['latency_ratio_mean']:.3f}", ""])
    p = R.write_tab(3, "tabA_t3_merge.tex", "lrrrr", ["Student", "mean KL", "p99 KL", "max KL", "argmax agree"], rows,
                    "% merged vs unmerged logits, 590 positions (first 4 GSM8K-train + 4 short-train prompts); bottom: "
                    "latency ratio vs target-only of the unmerged peft wrapper vs the merged checkpoint")
    R.record(p, S, [SUM])
    rows = []
    for g in (2, 4):
        for d, v in P["step6"][f"gamma{g}_diag"].items():
            rows.append([f"{d.replace('_', ' ')}, $\\gamma$={g}", f"{v['accepted_over_examined']:.3f}", f"{v['latency_ratio_mean']:.3f}"])
    p = R.write_tab(3, "tabA_t3_fixed64.tex", "lrr", ["Draft", "acc/exam.", "latency ratio"], rows,
                    "% first 24 GSM8K test IDs, 64 tokens with EOS ignored, 2 reps (timing only)")
    R.record(p, S, [SUM])
    t = R.load("results/task3/toy_chisquare.json")
    cor = {g: next(r for r in t["runs"] if r["gamma"] == g and r["sampler"] == "correct") for g in (1, 2, 3)}
    neg = next(r for r in t["runs"] if r["gamma"] == 2 and r["sampler"] != "correct")
    rows = []
    for i, j in enumerate(cor[1]["joint"]):
        rows.append(["".join(map(str, j["seq"])), f"{j['exact']:.4f}"] + [f"{cor[g]['joint'][i]['empirical']:.4f}" for g in (1, 2, 3)]
                    + [f"{neg['joint'][i]['empirical']:.4f}"])
    p = R.write_tab(3, "tabA_t3_chisq_joint.tex", "lrrrrr", ["seq", "exact", "$\\gamma$=1", "$\\gamma$=2", "$\\gamma$=3",
                                                             "control $\\gamma$=2"], rows,
                    "% exact joint law vs empirical frequency over 100,000 sequences")
    R.record(p, S, ["results/task3/toy_chisquare.json"])


def tabA_workload():
    """Speculative outcomes by workload and by realised output-length bin, every draft, both gamma."""
    rows = []
    eos64 = {}
    for g in (2, 4):
        rows.append([f"\\multicolumn{{6}}{{l}}{{\\textit{{$\\gamma$={g}}}}}"])
        for kind in ("workload", "length"):
            segs = ["math", "familiar", "shifted"] if kind == "workload" else ["[1,16]", "[17,32]", "[33,63]", "[64,64]"]
            for sg in segs:
                for k, d in enumerate(DRAFTS):
                    v = P["step6"][f"gamma{g}"][d]
                    if kind == "workload":
                        x = v["by_workload"][sg]
                        ci_, n = x["latency_ratio"], f"{x['n_runs']} ({x['n_prompts']})"
                        lab = sg
                    else:
                        x = v["by_output_length"].get(sg)
                        if x is None:
                            rows.append(["" if k else sg.strip("[]").replace(",", "--"), LAB[d], "", "", "", "0"])
                            continue
                        ci_, n = x["latency_ratio_runs"], f"{x['n_runs']} ({x['n_prompts']})"
                        eos64[(g, d)] = x["eos_at_64"] if sg == "[64,64]" else eos64.get((g, d), 0)
                        lab = "64 (cap)" if sg == "[64,64]" else sg.strip("[]").replace(",", "--")
                    rows.append([lab if k == 0 else "", LAB[d], f"{x['accepted_over_examined']:.3f}",
                                 f"{x['target_passes_per_token']:.3f}", R.ci(ci_, 1, "{:.2f}", signed=False), n])
                rows.append("MID")
    rows.pop()
    n64 = sum(eos64.values())
    note = ("% segment rows: workload (math / familiar / shifted), then realised output length in tokens (EOS-aware "
            "64-token cap). acc/exam. = accepted / examined draft tokens; passes/tok = target forward passes per "
            "generated token; latency ratio = spec / adjacent target-only. Workload: mean of per-prompt ratios "
            "(averaged over the 2 reps), 95% bootstrap CI over prompts. Length bins: mean over runs, 95% bootstrap CI "
            "clustered by prompt (a prompt's 2 reps can fall in different bins). n = runs (prompts). "
            f"The 64 bin includes {n64} run(s) that emitted EOS as the 64th token. All 5 drafts shown (no averaging).")
    p = R.write_tab(3, "tabA_t3_workload.tex", "llrrrr",
                    ["Segment", "Draft", "acc/exam.", "passes/tok", "latency ratio [95\\% CI]", "n runs (prompts)"],
                    rows, note)
    R.record(p, S, [SUM, "results/task3/step6_g2.jsonl", "results/task3/step6_g4.jsonl"])


def main():
    R.setup()
    tabA_workload()
    fig6()
    fig7()
    tab3_sampler()
    tab3_spec()
    tab3_costs()
    tabA_align()
    tabA_misc()
    R.dump_manifest(S)
    print("task3 outputs:", len(R.MANIFEST))


if __name__ == "__main__":
    main()
