#!/usr/bin/env python3
"""Task 2 report figures and table fragments (reads only results/task2/*). Outputs -> report/figs/task2/."""
from __future__ import annotations

import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import rstyle as R  # noqa: E402

S = "scripts/report/task2.py"
PA = "results/task2/partA"
PB = "results/task2/partB"
PC = "results/task2/partC"
OPS = {"dense": ("dense", R.OI["black"]), "tiled64": ("tiled-64", R.OI["orange"]), "tiled128": ("tiled-128", R.OI["green"]),
       "tiled256": ("tiled-256", R.OI["purple"]), "sdpa_default": ("SDPA", R.OI["blue"])}
MECH = {"dense": R.OI["black"], "window": R.OI["blue"], "linear": R.OI["green"], "dilated": R.OI["orange"],
        "curriculum": R.OI["verm"]}
KV = ["fp16", "sink_recent", "qaware", "int4_tok", "int4_chan", "hybrid"]
KVLAB = {"fp16": "FP16 ref", "sink_recent": "sink+recent", "qaware": "question-aware", "int4_tok": "INT4 token-wise",
         "int4_chan": "INT4 chanK/tokV", "hybrid": "hybrid", "int4_tok_zp": "INT4 token-wise, zero-point",
         "int4_chan_zp": "INT4 chanK/tokV, zero-point"}
KVCOL = {"fp16": R.OI["black"], "sink_recent": R.OI["orange"], "qaware": R.OI["blue"], "int4_tok": R.OI["verm"],
         "int4_chan": R.OI["purple"], "hybrid": R.OI["green"]}


# ---------------------------------------------------------------- Part A
def fig3():
    w, h = 5.5, 1.75
    tim = R.load(f"{PA}/timing.jsonl")
    cor = R.load(f"{PA}/correctness.json")["grid"]
    f = R.fig(w, h)
    a, b, c = f.subplots(1, 3)
    for op, (lab, col) in OPS.items():
        for dt, ls in (("fp16", "-"), ("fp32", "--")):
            rs = sorted([r for r in tim if r["op"] == op and r["dtype"] == dt and r["mode"] == "forward_backward"
                         and r["status"] == "ok"], key=lambda r: r["L"])
            if rs:
                a.plot([r["L"] for r in rs], [r["median"] for r in rs], ls=ls, color=col, lw=0.8, marker="o", ms=1.8)
                b.plot([r["L"] for r in rs], [r["max_memory_allocated_bytes"] / 2 ** 20 for r in rs], ls=ls, color=col,
                       lw=0.8, marker="o", ms=1.8)
        rs = sorted([r for r in cor if r["op"] == op and r["dtype"] == "fp16" and r["status"] == "ok"], key=lambda r: r["L"])
        c.plot([r["L"] for r in rs], [r["out"]["max_abs"] for r in rs], color=col, marker="o", ms=2.4, lw=0.6)
        c.plot([r["L"] for r in rs], [max(r[k]["max_abs"] for k in ("dq", "dk", "dv")) for r in rs], color=col,
               marker="o", ms=2.4, mfc="none", lw=0.6, ls=":")
    for ax in (a, b, c):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length $L$")
        ax.set_xticks([256, 1024, 4096], ["256", "1k", "4k"])
    a.set_ylabel("fwd+bwd latency (ms)")
    b.set_ylabel("peak allocated (MiB)")
    c.set_ylabel("max |error| vs FP32 ref. (FP16)")
    hs = [Line2D([], [], color=col, lw=0.9) for _, col in OPS.values()]
    hs += [Line2D([], [], color="k", ls="-", lw=0.8), Line2D([], [], color="k", ls="--", lw=0.8),
           Line2D([], [], color="k", marker="o", ls="none", ms=2.4), Line2D([], [], color="k", marker="o", mfc="none",
                                                                         ls="none", ms=2.4)]
    f.legend(hs, [l for l, _ in OPS.values()] + ["FP16", "FP32", "output (c)", "grad. (c)"], loc="outside lower center",
             ncol=9, fontsize=R.SMALL, columnspacing=0.8, handlelength=1.3)
    for ax, l in zip((a, b, c), "abc"):
        R.panel(ax, l)
    R.record(R.save(f, 2, "fig3_attention_ops.pdf", w, h), S, [f"{PA}/timing.jsonl", f"{PA}/correctness.json"])


def tab2_backends():
    bt = R.load(f"{PA}/backend_table.json")
    names = ["FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"]
    rows = []
    for be in names:
        cells = []
        for dt in ("fp16", "fp32"):
            for mode in ("forward", "forward_backward"):
                ok = [r[mode]["supported"] for r in bt["rows"] if r["backend"] == be and r["dtype"] == dt]
                cells.append("\\checkmark" if all(ok) else ("$\\times$" if not any(ok) else "partial"))
        rows.append([be.split("_")[0].title() if be != "MATH" else "Math"] + cells)
    disp = sorted({d["fused_sdp_choice"] for d in bt["default_dispatch"]})
    rows.append("MID")
    rows.append([f"\\multicolumn{{5}}{{l}}{{default dispatch (all $L$, both dtypes): "
                 f"{', '.join(x.split('_')[0].title() for x in disp)}}}"])
    p = R.write_tab(2, "tab2_backends.tex", "lcccc", ["Backend", "FP16 fwd", "FP16 f+b", "FP32 fwd", "FP32 f+b"], rows,
                    "% Tesla T4 (sm75), torch 2.11; L in {256..8192}")
    R.record(p, S, [f"{PA}/backend_table.json"])


def tabA_partA():
    tim = R.load(f"{PA}/timing.jsonl")
    cor = {(r["L"], r["dtype"], r["op"]): r for r in R.load(f"{PA}/correctness.json")["grid"]}
    rows = []
    for dt in ("fp16", "fp32"):
        for L in (256, 512, 1024, 2048, 4096, 8192):
            for op, (lab, _) in OPS.items():
                tf = next(r for r in tim if r["L"] == L and r["dtype"] == dt and r["op"] == op and r["mode"] == "forward")
                tb = next(r for r in tim if r["L"] == L and r["dtype"] == dt and r["op"] == op and r["mode"] == "forward_backward")
                c = cor[(L, dt, op)]
                if c["status"] != "ok":
                    rows.append([dt, str(L), lab, c["status"], "", "", "", "", ""])
                    continue
                g = max(c[k]["max_abs"] for k in ("dq", "dk", "dv"))
                rows.append([dt.upper(), str(L), lab, f"{c['out']['max_abs']:.1e}", f"{g:.1e}",
                             f"{tf['median']:.2f}" if tf["status"] == "ok" else tf["status"],
                             f"{tb['median']:.2f}" if tb["status"] == "ok" else tb["status"],
                             f"{tf.get('max_memory_allocated_bytes', 0) / 2 ** 20:.0f}",
                             f"{tb.get('max_memory_allocated_bytes', 0) / 2 ** 20:.0f}"])
        rows.append("MID")
    rows.pop()
    p = R.write_tab(2, "tabA_t2_partA_full.tex", "lrlrrrrrr",
                    ["dtype", "$L$", "Operator", "out err", "grad err", "fwd ms", "f+b ms", "fwd MiB", "f+b MiB"], rows,
                    "% no OOM / unsupported rows occurred for these operators; CPU stream generation ran concurrently "
                    "(deviations.md)")
    R.record(p, S, [f"{PA}/timing.jsonl", f"{PA}/correctness.json"])


# ---------------------------------------------------------------- Part B
GROUP_ROWS = [("dense_K4_mixed", "dense K4 mixed"), ("dense_K8_mixed", "dense K8 mixed"),
              ("dense_K16_mixed", "dense K16 mixed"), ("MASK", "dense K16 + test W=64 mask"),
              ("dense_K16_near", "dense K16 near"), ("window_K4_near", "window K4 near"),
              ("window_K8_near", "window K8 near"), ("window_K16_near", "window K16 near"),
              ("window_K16_mixed", "window K16 mixed"), ("window_K16_curriculum", "B3 curriculum (window K16)"),
              ("linear_K4_mixed", "linear K4 mixed"), ("linear_K8_mixed", "linear K8 mixed"),
              ("linear_K16_mixed", "linear K16 mixed"), ("dilated_K4_mixed", "dilated K4 mixed"),
              ("dilated_K16_mixed", "dilated K16 mixed")]
CELLS = [f"K{k}_{r}" for k in (4, 8, 16) for r in ("near", "far", "mixed")]


def mask_control():
    vals = defaultdict(list)
    for s in (6304, 6305, 6306):
        r = R.load(f"{PB}/runs/dense_K16_mixed_s{s}.json")
        for c, v in r["test_with_window_mask_at_test_time"].items():
            vals[c].append(v["acc"])
    return {c: st.mean(v) for c, v in vals.items()}


def trained_cell(g):
    if g == "MASK":
        return "K16_mixed"
    parts = g.split("_")
    return f"{parts[1]}_{'mixed' if parts[2] == 'curriculum' else parts[2]}"


def fig4():
    w, h = 5.5, 2.45
    G = R.load(f"{PB}/summary.json")["groups"]
    mc = mask_control()
    f = R.fig(w, h)
    gs = f.add_gridspec(2, 2, width_ratios=[3.1, 2.4])
    a = f.add_subplot(gs[:, 0])
    b = f.add_subplot(gs[0, 1])
    c = f.add_subplot(gs[1, 1])
    M = np.array([[mc[c_] if g == "MASK" else G[g]["test"][c_]["mean"] for c_ in CELLS] for g, _ in GROUP_ROWS])
    a.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4.6,
                   color="white" if M[i, j] > 0.6 else "black")
        j = CELLS.index(trained_cell(GROUP_ROWS[i][0]))
        a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=R.OI["verm"], lw=0.9))
    a.set_xticks(range(9), [c_.replace("_", " ") for c_ in CELLS], rotation=60, ha="right", fontsize=5.5)
    a.set_yticks(range(len(GROUP_ROWS)), [l for _, l in GROUP_ROWS], fontsize=5.5)
    a.tick_params(length=0)
    lsK = {4: ":", 8: "--", 16: "-"}
    for g, v in G.items():
        arch, K = g.split("_")[0], int(g.split("_")[1][1:])
        mech = "curriculum" if "curriculum" in g else arch
        xs = [int(u) for u in v["dev_curve_trained_cell"]]
        mean = [v["dev_curve_trained_cell"][str(u)]["mean"] for u in xs]
        lo = [min(v["dev_curve_trained_cell"][str(u)]["values"]) for u in xs]
        hi = [max(v["dev_curve_trained_cell"][str(u)]["values"]) for u in xs]
        b.plot(xs, mean, color=MECH[mech], ls=lsK[K], lw=0.8)
        b.fill_between(xs, lo, hi, color=MECH[mech], alpha=0.12, lw=0)
    b.set_xlabel("update")
    b.set_ylabel("dev acc. (trained cell)")
    b.set_ylim(0, 1.5)
    b.set_yticks([0, 0.5, 1.0])
    hs = [Line2D([], [], color=col, lw=1) for col in MECH.values()] + [Line2D([], [], color="k", ls=l, lw=0.8)
                                                                        for l in lsK.values()]
    b.legend(hs, list(MECH) + [f"K={k}" for k in lsK], ncol=4, fontsize=5.0, loc="upper center", columnspacing=0.5,
             handlelength=1.3, borderaxespad=0.1)
    bins = ["[0,32)", "[32,64)", "[64,128)", "[128,256)"]
    mids = [16, 48, 96, 192]
    sel = [("dense_K16_mixed", "dense mixed", MECH["dense"], "-"), ("dense_K16_near", "dense near", MECH["dense"], ":"),
           ("window_K16_near", "window near", MECH["window"], ":"), ("window_K16_mixed", "window mixed", MECH["window"], "-"),
           ("window_K16_curriculum", "curriculum", MECH["curriculum"], "-"), ("linear_K16_mixed", "linear", MECH["linear"], "-"),
           ("dilated_K16_mixed", "dilated", MECH["dilated"], "-")]
    for g, lab, col, ls in sel:
        ys = [G[g]["dist_bins"]["K16_mixed"][bb]["mean"] for bb in bins]
        c.plot(mids, ys, color=col, ls=ls, marker="o", ms=2, lw=0.8, label=lab)
    c.axvline(126, color=R.OI["grey"], lw=0.6, ls="--")
    c.text(118, 0.25, "126", fontsize=5.5, color=R.OI["grey"], ha="right")
    c.set_xscale("log", base=2)
    c.set_xticks(mids, ["0-32", "32-64", "64-128", "128-256"], fontsize=5.5)
    c.set_xlabel("query-source distance (mixed K=16 test)")
    c.set_ylabel("accuracy")
    c.set_ylim(-0.05, 1.55)
    c.set_yticks([0, 0.5, 1.0])
    c.legend(ncol=4, fontsize=5.0, loc="upper center", handlelength=1.2, columnspacing=0.5, borderaxespad=0.1)
    R.panel(a, "a", dx=-0.02)
    R.panel(b, "b")
    R.panel(c, "c")
    R.record(R.save(f, 2, "fig4_mqar.pdf", w, h), S, [f"{PB}/summary.json"] + [f"{PB}/runs/dense_K16_mixed_s{s}.json"
                                                                             for s in (6304, 6305, 6306)])


def tab2_mqar_resources():
    ts = R.load(f"{PB}/train_step_timing.json")
    sw = R.load(f"{PB}/inference_sweep.jsonl")
    th, sb = ts["theory"], ts["state_bytes_T256_fp32"]
    spec = [("dense", "dense", "dense_sdpa_causal", th["pairs_total_per_head"]["dense"], sb["dense_kv"], "no"),
            ("window", "window $W$=64", "window_dense_mask", th["pairs_total_per_head"]["window"], sb["window_kv"], "dense mask"),
            ("dilated", "local+dilated", "dilated_mixed_dense_mask", th["pairs_total_per_head"]["mixed_local_dilated_mean_over_heads"],
             sb["dilated_mixed_kv"], "dense mask"),
            ("linear", "linear (cumsum)", "linear_cumsum", None, sb["linear_S_z"], "prefix tensors"),
            ("linear", "linear (recurrent)", "linear_recurrent", None, sb["linear_S_z"], "no")]
    rows = []
    for arch, lab, op, pairs, state, mat in spec:
        tr = ts["per_arch"][arch]
        inf = next(r for r in sw if r["op"] == op and r["T"] == 8192)
        rows.append([lab, f"{pairs / 256:.1f}" if pairs else "$O(1)$ state", f"{state / 1024:.1f}",
                     f"{tr['median']:.1f}" if "recurrent" not in lab else "", f"{tr['max_memory_allocated_bytes'] / 2 ** 20:.0f}"
                     if "recurrent" not in lab else "", f"{inf['median']:.1f}", f"{inf['max_memory_allocated_bytes'] / 2 ** 20:.0f}", mat])
    p = R.write_tab(2, "tab2_mqar_resources.tex", "lrrrrrrl",
                    ["Operator", "pairs/query", "state KiB", "step ms", "train MiB", "inf. ms", "inf. MiB", "materialises"],
                    rows, "% pairs/query = attended pairs per head / T at T=256; decode state per layer, FP32, T=256; "
                          "train step: exclusive, B=32, T=256; inference: forward only, T=8192, B=4, FP32")
    R.record(p, S, [f"{PB}/train_step_timing.json", f"{PB}/inference_sweep.jsonl"])


def tabA_mqar_full():
    G = R.load(f"{PB}/summary.json")["groups"]
    rows = []
    for g, lab in GROUP_ROWS:
        if g == "MASK":
            continue
        t = G[g]["test"]
        rows.append([lab] + [f"{t[c]['mean']:.3f}$\\pm${t[c]['sd']:.3f}" for c in CELLS])
    p = R.write_tab(2, "tabA_t2_mqar_full.tex", "l" + "r" * 9, ["Group"] + [c.replace("_", " ") for c in CELLS], rows,
                    "% mean +/- sd over seeds 6304/6305/6306; LANDSCAPE recommended")
    R.record(p, S, [f"{PB}/summary.json"])
    rows = []
    bins = ["[0,32)", "[32,64)", "[64,128)", "[128,256)"]
    for g, lab in GROUP_ROWS:
        if g == "MASK":
            continue
        rows.append([lab] + [f"{G[g]['dist_bins']['K16_mixed'][b]['mean']:.3f}" for b in bins])
    p = R.write_tab(2, "tabA_t2_mqar_bins.tex", "lrrrr", ["Group"] + bins, rows,
                    "% accuracy on the mixed K=16 test cell by query-source distance; mean over 3 seeds")
    R.record(p, S, [f"{PB}/summary.json"])


def tabA_dilated():
    d = R.load(f"{PB}/dilated_alignment.json")
    rows = []
    for g in ("dilated_K4_mixed", "dilated_K16_mixed", "dense_K4_mixed", "dense_K16_mixed"):
        K = g.split("_")[1]
        p = d["pooled"][g][f"trained_cell_{K}_mixed"]
        for kind, lab in (("value_offset_mod4", "value"), ("key_offset_mod4", "key")):
            for band, bl in (("near_lt128", "$<$128"), ("far_ge128", "$\\geq$128")):
                rows.append([g.replace("_", " "), lab, bl] + [f"{p[kind][band][r]['acc']:.3f}" for r in "0123"])
    p = R.write_tab(2, "tabA_t2_dilated.tex", "lllrrrr", ["Run group", "offset", "distance", "r=0", "r=1", "r=2", "r=3"],
                    rows, "% accuracy by offset residue mod 4 on the trained cell, pooled over 3 seeds")
    R.record(p, S, [f"{PB}/dilated_alignment.json"])


def figA_curriculum():
    w, h = 5.5, 1.5
    G = R.load(f"{PB}/summary.json")["groups"]
    f = R.fig(w, h)
    a, b = f.subplots(1, 2)
    for ax in (a, b):
        ax.axvspan(0, 1500, color=R.OI["sky"], alpha=0.12, lw=0)
        ax.axvspan(1500, 3000, color=R.OI["yellow"], alpha=0.15, lw=0)
    a.text(750, 1.02, "near only", ha="center", fontsize=5.5)
    a.text(2250, 1.02, "ramp", ha="center", fontsize=5.5)
    a.text(3750, 1.02, "mixed", ha="center", fontsize=5.5)
    for g, lab, col in (("window_K16_curriculum", "curriculum", MECH["curriculum"]),
                        ("window_K16_mixed", "window mixed", MECH["window"]),
                        ("window_K16_near", "window near", R.OI["sky"])):
        v = G[g]["dev_curve_trained_cell"]
        xs = [int(u) for u in v]
        a.plot(xs, [v[str(u)]["mean"] for u in xs], color=col, marker="o", ms=2, lw=0.8, label=lab)
        a.fill_between(xs, [min(v[str(u)]["values"]) for u in xs], [max(v[str(u)]["values"]) for u in xs], color=col,
                       alpha=0.15, lw=0)
    a.set_xlabel("update")
    a.set_ylabel("dev acc. (own trained cell)")
    a.set_ylim(0, 1.1)
    a.legend(fontsize=5.5, loc="center right")
    for s in (6304, 6305, 6306):
        r = R.load(f"{PB}/runs/window_K16_curriculum_s{s}.json")
        xs = [d["update"] for d in r["dev_curve"]]
        for cell, col in (("K16_near", R.OI["blue"]), ("K16_mixed", R.OI["verm"]), ("K16_far", R.OI["grey"])):
            b.plot(xs, [d["dev"][cell]["acc"] for d in r["dev_curve"]], color=col, lw=0.6, alpha=0.8,
                   label=cell.replace("_", " ") if s == 6304 else None)
    b.axhline(0.501, color=R.OI["verm"], ls=":", lw=0.6)
    b.set_xlabel("update")
    b.set_ylabel("curriculum dev acc.")
    b.set_ylim(0, 1.1)
    b.legend(fontsize=5.5, loc="center right")
    R.panel(a, "a")
    R.panel(b, "b")
    R.record(R.save(f, 2, "figA_t2_curriculum.pdf", w, h), S, [f"{PB}/summary.json"] +
             [f"{PB}/runs/window_K16_curriculum_s{s}.json" for s in (6304, 6305, 6306)])


# ---------------------------------------------------------------- Part C
def qrows():
    rows = R.load(f"{PC}/quality_s0.jsonl") + R.load(f"{PC}/quality_s1.jsonl")
    return rows


def tab2_kv():
    P = R.load(f"{PC}/summary.json")
    T, Cn = P["table"], P["contrasts"]
    rows = []
    for m in KV + ["MID", "int4_tok_zp", "int4_chan_zp"]:
        if m == "MID":
            rows.append("MID")
            continue
        t = T[m]
        if m == "fp16":
            acc = f"{t['accuracy']:.3f}"
        else:
            key = f"{m}-fp16" + (" [supplementary: zero-point sensitivity]" if m.endswith("_zp") else "")
            c = Cn[key]["accuracy"]
            acc = f"{t['accuracy']:.3f} [{t['accuracy'] - c['point'] + c['ci_low']:.3f}, " \
                  f"{t['accuracy'] - c['point'] + c['ci_high']:.3f}]"
        rec = t.get("recon_rel_fro")
        rows.append([KVLAB[m] + ("$^\\ast$" if m.endswith("_zp") else ""), acc, f"{t['ref_answer_nll_mean']:.3f}",
                     f"{t['kl_first_token_mean']:.3f}", "" if t["evidence_retained_fp16_mean"] is None else
                     f"{t['evidence_retained_fp16_mean']:.3f}", f"{100 * t['fraction_of_M16']['mean']:.2f}",
                     f"{rec['K']:.3f} / {rec['V']:.3f}" if rec else ""])
    p = R.write_tab(2, "tab2_kv.tex", "lrrrrrr",
                    ["Cache", "Accuracy [95\\% CI]", "Ref. NLL", "KL", "Evid.", "\\% FP16 B", "K / V err."], rows,
                    "% 180 prompts; accuracy CI = FP16 accuracy (1.0) + paired bootstrap CI of the difference to FP16; "
                    "* supplementary: zero-point sensitivity (not one of the five required caches)")
    R.record(p, S, [f"{PC}/summary.json"])


def tab2_kv_contrasts():
    Cn = R.load(f"{PC}/summary.json")["contrasts"]
    spec = [("question-aware $-$ sink+recent", "qaware-sink_recent"), ("chanK/tokV $-$ token-wise", "int4_chan-int4_tok"),
            ("hybrid $-$ question-aware", "hybrid-qaware"), ("hybrid $-$ chanK/tokV", "hybrid-int4_chan"),
            ("sink+recent $-$ FP16", "sink_recent-fp16"), ("hybrid $-$ FP16", "hybrid-fp16")]
    rows = [[lab, R.ci(Cn[k]["accuracy"], 1, "{:.3f}"), R.ci(Cn[k]["ref_answer_nll"], 1, "{:.3f}")] for lab, k in spec]
    p = R.write_tab(2, "tab2_kv_contrasts.tex", "lrr", ["Contrast", "$\\Delta$accuracy [95\\% CI]",
                                                        "$\\Delta$ref. NLL [95\\% CI]"], rows,
                    "% paired bootstrap over 180 prompts, 2,000 resamples, seed 6304")
    R.record(p, S, [f"{PC}/summary.json"])


def fig5():
    w, h = 5.5, 2.3
    P = R.load(f"{PC}/summary.json")
    rows = qrows()
    f = R.fig(w, h)
    gs = f.add_gridspec(2, 3, height_ratios=[3.3, 1])
    a, b, c = f.add_subplot(gs[0, 0]), f.add_subplot(gs[0, 1]), f.add_subplot(gs[0, 2])
    m_ax = f.add_subplot(gs[1, :])
    pos = ["start", "middle", "end"]
    pols = [m for m in KV if m != "fp16"]
    wbar = 0.8 / len(pols)
    for i, m in enumerate(pols):
        ys = [P["breakdown"]["evidence_position"][p_][m]["accuracy"] for p_ in pos]
        a.bar(np.arange(3) + (i - 2) * wbar, ys, width=wbar, color=KVCOL[m], label=KVLAB[m])
    a.set_xticks(range(3), pos)
    a.set_ylabel("accuracy")
    a.set_ylim(0, 1.05)
    for m in KV:
        ys = []
        for d in (0, 2, 8):
            vals = [r["methods"][m]["ref_nll_sum"] for r in rows if r["cell"][2] == d]
            ys.append(st.mean(vals))
        b.plot([0, 2, 8], ys, color=KVCOL[m], marker="o", ms=2.2, lw=0.8)
    b.set_yscale("log")
    b.set_xticks([0, 2, 8])
    b.set_xlabel("distractors")
    b.set_ylabel("ref.-answer NLL (nats)")
    pb = [r for r in P["packed_bench"]["rows"] if r["kind"] == "decode_step" and r["L_tag"] == "actual"]
    labs, x = [], 0
    for r in sorted(pb, key=lambda r: (r["B"], r["L"])):
        c.bar(x - 0.18, r["fp16_cache_step_ms"]["median"], width=0.36, color=R.OI["black"])
        c.bar(x + 0.18, r["int4_chan_unpack_dequant_plus_step_ms"]["median"], width=0.36, color=R.OI["purple"])
        labs.append(f"B{r['B']}\n{r['L'] / 1000:.1f}k")
        x += 1
    c.set_xticks(range(len(labs)), labs, fontsize=5.5)
    c.set_ylabel("decode step (ms)")
    c.legend([Rectangle((0, 0), 1, 1, color=R.OI["black"]), Rectangle((0, 0), 1, 1, color=R.OI["purple"])],
             ["FP16 cache", "INT4 unpack+dequant+step"], fontsize=5.2, loc="upper left")
    ex = next(r for r in rows if tuple(r["cell"]) == (2048, "middle", 8))
    L = ex["L"]
    for k, (m, lab) in enumerate((("sink_recent", "sink+recent"), ("qaware", "question-aware"), ("hybrid", "hybrid FP16"))):
        for s0, n in ex["methods"][m]["fp16_selection_rle"]:
            m_ax.add_patch(Rectangle((s0, 2 - k - 0.4), n, 0.8, color=KVCOL[m], lw=0))
    e0, e1 = ex["evidence_token_range"]
    m_ax.axvspan(e0, e1, color=R.OI["yellow"], alpha=0.6, lw=0)
    m_ax.set_xlim(0, L)
    m_ax.set_ylim(-0.6, 2.6)
    m_ax.set_yticks([2, 1, 0], ["sink+recent", "question-aware", "hybrid FP16"], fontsize=5.5)
    m_ax.set_xlabel(f"context position ({ex['id']}, $L$={L}; gold evidence shaded)")
    hs = [Rectangle((0, 0), 1, 1, color=KVCOL[m]) for m in pols] + [Line2D([], [], color=KVCOL["fp16"], marker="o", ms=2)]
    f.legend(hs, [KVLAB[m] for m in pols] + ["FP16 ref"], loc="outside upper center", ncol=6, fontsize=R.SMALL,
             columnspacing=0.8, handlelength=1.0)
    for ax, l in zip((a, b, c, m_ax), "abcd"):
        R.panel(ax, l)
    R.record(R.save(f, 2, "fig5_kv.pdf", w, h), S, [f"{PC}/summary.json", f"{PC}/quality_s0.jsonl", f"{PC}/quality_s1.jsonl"])


def tabA_kv_breakdown():
    rows = qrows()
    out = []
    for T in (2048, 4096):
        for pos in ("start", "middle", "end"):
            for d in (0, 2, 8):
                rs = [r for r in rows if tuple(r["cell"]) == (T, pos, d)]
                out.append([str(T), pos, str(d)] + [f"{st.mean(r['methods'][m]['correct'] for r in rs):.1f}" for m in KV[1:]])
    p = R.write_tab(2, "tabA_t2_kv_breakdown.tex", "llr" + "r" * 5,
                    ["Len.", "Evid.", "Distr."] + [KVLAB[m] for m in KV[1:]], out,
                    "% accuracy per 10-prompt cell (descriptive); FP16 reference is 1.0 in every cell")
    R.record(p, S, [f"{PC}/quality_s0.jsonl", f"{PC}/quality_s1.jsonl"])


def tabA_kv_generations():
    P = R.load(f"{PC}/summary.json")
    rows = []
    for g in P["aligned_generations"]:
        rows.append([f"\\multicolumn{{3}}{{@{{}}l@{{}}}}{{\\parbox{{4.0in}}{{\\textit{{{g['id']}}} ($L$={g['L']}); evidence: "
                     f"{R_esc(g['evidence_text'])}}}}}"])
        for m in KV + ["int4_tok_zp", "int4_chan_zp"]:
            ev = g["evidence_retained"].get(m)
            rows.append([KVLAB[m], "\\texttt{" + R_esc(g["outputs"][m][:40]) + "}" + (" \\checkmark" if g["correct"][m] else ""),
                         "" if ev is None else f"{ev:.2f}"])
        rows.append("MID")
    rows.pop()
    p = R.write_tab(2, "tabA_t2_kv_generations.tex", "llr", ["Cache", "Output (first 40 chars)", "Evid."], rows,
                    "% rule: first prompt ID of cells (2048,middle,8), (4096,start,8), (4096,middle,2)")
    R.record(p, S, [f"{PC}/summary.json"])


def R_esc(s):
    return s.replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#") \
        .replace("$", "\\$").replace("{", "").replace("}", "").replace("\n", " ")


def tabA_packed():
    P = R.load(f"{PC}/summary.json")["packed_bench"]["rows"]
    rows = []
    for r in P:
        if r["kind"] == "decode_step":
            continue
        step = next(x for x in P if x["kind"] == "decode_step" and x["L"] == r["L"] and x["B"] == r["B"])
        rows.append([f"{r['L']} ({r['L_tag']})", str(r["B"]), r["kind"].replace("_", " "), f"{r['encode_quantize_pack_ms']['median']:.1f}",
                     f"{r['decode_unpack_dequant_ms']['median']:.1f}", f"{r['serialized_quantized_part_bytes'] / 2 ** 20:.1f}",
                     f"{r['fp16_same_tokens_bytes'] / 2 ** 20:.1f}",
                     f"{step['fp16_cache_step_ms']['median']:.1f} / {step['int4_chan_unpack_dequant_plus_step_ms']['median']:.1f}"])
    p = R.write_tab(2, "tabA_t2_packed.tex", "lrlrrrrr",
                    ["$L$", "B", "Format", "enc ms", "dec ms", "packed MiB", "FP16 MiB", "step ms FP16 / unpack"], rows,
                    "% median of 10 warmed, synced reps; packed = serialized quantized part only; step: INT4 chanK/tokV")
    R.record(p, S, [f"{PC}/summary.json"])


def tabA_kquant():
    d = R.load(f"{PC}/k_quant_diagnostics.json")
    cmp_ = R.load(f"{PC}/k_quant_convention_compare.json")
    rows = []
    for pid, r in d["prompts"].items():
        for key, lab in (("old/channel", "clamped zero-point (bug), channel"), ("old/token", "clamped zero-point (bug), token"),
                         ("fixed/channel", "min/offset, channel"), ("fixed/token", "min/offset, token"),
                         ("exact_fp32_meta/channel", "min/offset FP32 meta, channel")):
            post, pre = r["post_rope_K"][key], r["pre_rope_K"][key]
            rows.append([pid, lab, f"{post['rel_mean_over_layers']:.3f}", f"{post['rel_all_layers']:.3f}",
                         f"{post['share_of_sq_error_in_first_group']:.3f}", f"{post['top10_channels_share_of_sq_error']:.2f}",
                         f"{pre['rel_mean_over_layers']:.3f}"])
        c = cmp_[pid]
        rows.append([pid, "zero-point unclamped, channel / token", f"{c['K_post/channel/zp_unclamped']['mean_over_layers']:.3f} / "
                     f"{c['K_post/token/zp_unclamped']['mean_over_layers']:.3f}", "", "", "", ""])
        rows.append("MID")
    rows.pop()
    p = R.write_tab(2, "tabA_t2_kquant.tex", "llrrrrr",
                    ["Prompt", "Quantizer, K layout", "rel. err (mean L)", "pooled", "1st-group share", "top-10 ch.",
                     "pre-RoPE"], rows, "% INT4 group-64 K reconstruction error on real cached keys; LANDSCAPE or "
                                        "appendix full width")
    R.record(p, S, [f"{PC}/k_quant_diagnostics.json", f"{PC}/k_quant_convention_compare.json"])


def main():
    R.setup()
    fig3()
    tab2_backends()
    tabA_partA()
    fig4()
    tab2_mqar_resources()
    tabA_mqar_full()
    tabA_dilated()
    figA_curriculum()
    tab2_kv()
    tab2_kv_contrasts()
    fig5()
    tabA_kv_breakdown()
    tabA_kv_generations()
    tabA_packed()
    tabA_kquant()
    R.dump_manifest(S)
    print("task2 outputs:", len(R.MANIFEST))


if __name__ == "__main__":
    main()
