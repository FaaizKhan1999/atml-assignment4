#!/usr/bin/env python3
"""Task 1 report figures and table fragments (reads only results/task1/*). Outputs -> report/figs/task1/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import rstyle as R  # noqa: E402

S = "scripts/report/task1.py"
SUM = "results/task1/summary_all.json"
C = R.load(SUM)["conditions"]
CT = R.load(SUM)["contrasts"]
BASE = C["baseline"]
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PSHORT = ["q", "k", "v", "o", "gate", "up", "down"]
# fixed method -> colour / marker map for every Task 1 figure
STYLE = {"P1": (R.OI["orange"], "o"), "P2": (R.OI["blue"], "o"), "P3": (R.OI["green"], "o"),
         "2:4": (R.OI["sky"], "D"), "Q1": (R.OI["verm"], "s"), "Q2": (R.OI["purple"], "s"),
         "Q3-proj": (R.OI["yellow"], "^"), "FP16": (R.OI["black"], "*")}
GB = 1e9


def family(c):
    return {"baseline": "FP16", "p2_wanda_2of4": "2:4"}.get(c) or \
        ("P1" if c.startswith("p1") else "P2" if c.startswith("p2") else "P3" if c.startswith("p3") else
         "Q1" if c.startswith("q1") else "Q2" if c.startswith("q2") else "Q3-proj" if c.startswith("q3proj") else "Q3")


def total_gb(c):
    st = C[c]["storage"]
    if c == "baseline":
        return st["dense_fp16_total_bytes"] / GB
    if c == "p2_wanda_2of4":
        return st["two_four_total_bytes"] / GB
    if st["quant_total_bytes"]:
        return st["quant_total_bytes"] / GB
    return st["sparse16_total_bytes"] / GB


def series():
    out = {"P1": ["p1_mag_s25", "p1_mag_s50", "p1_mag_s75"], "P2": ["p2_wanda_s25", "p2_wanda_s50", "p2_wanda_s75"],
           "P3": ["p3_owl_s50", "p3_owl_s75"], "2:4": ["p2_wanda_2of4"], "Q1": ["q1_int8", "q1_int4", "q1_int3"],
           "Q2": ["q2_awq_int8", "q2_awq_int4", "q2_awq_int3"], "Q3-proj": ["q3proj_b4", "q3proj_b3"],
           "FP16": ["baseline"]}
    return out


def mk(ax, fam, xs, ys, line, ms=3.2, open_mask=None):
    col, m = STYLE[fam]
    face = col
    ec = "black" if fam in ("Q3-proj",) else col
    if line and len(xs) > 1:
        ax.plot(xs, ys, color=col, lw=0.8, zorder=2)
    for i, (x, y) in enumerate(zip(xs, ys)):
        op = open_mask[i] if open_mask else False
        ax.plot([x], [y], marker=m, ms=ms + (1.5 if fam == "FP16" else 0), color=col, mfc="none" if op else face,
                mec=ec, mew=0.6, ls="none", zorder=3)


def fig1():
    w, h = 5.5, 1.85
    f = R.fig(w, h)
    axs = f.subplots(1, 4)
    ser = series()
    for fam, cs in ser.items():
        xs = [total_gb(c) for c in cs]
        line = fam in ("P1", "P2", "P3")
        mk(axs[0], fam, xs, [C[c]["wikitext2"]["ppl"] for c in cs], line)
        mk(axs[1], fam, xs, [C[c]["hellaswag"]["first_512"]["acc_norm"] for c in cs], line)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("total model bytes (GB)")
    axs[0].set_ylabel("WikiText-2 PPL")
    axs[1].set_xlabel("total model bytes (GB)")
    axs[1].set_ylabel("HellaSwag acc_norm (first 512)")
    axs[1].axhline(0.25, color=R.OI["grey"], lw=0.5, ls=":")
    b = BASE["wikitext2"]["mean_nll"]
    floor = 1e-4
    for fam, cs in ser.items():
        cs = [c for c in cs if c != "baseline"]
        if not cs:
            continue
        xs = [C[c]["output_error_param_weighted"] for c in cs]
        d = [C[c]["wikitext2"]["mean_nll"] - b for c in cs]
        mk(axs[2], fam, xs, [max(v, floor) for v in d], False, open_mask=[v <= 0 for v in d])
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("calib. output error (param-wt.)")
    axs[2].set_ylabel(r"WikiText-2 $\Delta$NLL (nats)")
    axs[2].text(0.03, 0.97, "open: $\\Delta$NLL$\\leq$0\n(shown at $10^{-4}$)", transform=axs[2].transAxes,
                fontsize=5.5, va="top")
    ps = R.load("results/task1/pruning_stats.json")["per_projection"]
    sp = [0.25, 0.5, 0.75]
    cols = [R.OI[k] for k in ("blue", "orange", "green", "verm", "purple", "sky", "black")]
    for i, (p, s) in enumerate(zip(PROJ, PSHORT)):
        ys = [np.mean([x["jaccard_pruned_mag_vs_wanda"][str(v)] for x in ps if x["name"] == p]) for v in sp]
        axs[3].plot([100 * v for v in sp], ys, marker="o", ms=2.2, lw=0.8, color=cols[i], label=s)
    axs[3].set_xlabel("sparsity (%)")
    axs[3].set_ylabel("Jaccard(magnitude, Wanda)")
    axs[3].set_xticks([25, 50, 75])
    axs[3].set_ylim(0.62, 0.93)
    axs[3].legend(ncol=4, loc="lower right", fontsize=5.5, columnspacing=0.5, handlelength=0.9)
    for a, l in zip(axs, "abcd"):
        R.panel(a, l)
    handles = [Line2D([], [], color=STYLE[k][0], marker=STYLE[k][1], ls="none", ms=3.5,
                      mec="black" if k == "Q3-proj" else STYLE[k][0]) for k in STYLE]
    f.legend(handles, list(STYLE), loc="outside lower center", ncol=8, fontsize=SMALLF, handletextpad=0.2,
             columnspacing=0.8)
    R.record(R.save(f, 1, "fig1_tradeoffs.pdf", w, h), S, [SUM, "results/task1/pruning_stats.json"])


SMALLF = 6.5


def fig2():
    w, h = 5.5, 2.0
    g = R.load("results/task1/diagnostics.json")
    qp = R.load("results/task1/q3proj_allocation.json")
    f = R.fig(w, h)
    gs = f.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
    a = f.add_subplot(gs[:, 0])
    b = f.add_subplot(gs[:, 1])
    c1, c2 = f.add_subplot(gs[0, 2]), f.add_subplot(gs[1, 2])
    L = 28
    x = np.arange(L)
    ax2 = a.twinx()
    from matplotlib.patches import Patch
    bar_handles = []
    for k, (S_, col) in enumerate((("0.5", R.OI["blue"]), ("0.75", R.OI["verm"]))):
        p = g["p3_blocks"][S_]
        ax2.bar(x + (k - 0.5) * 0.4, np.array(p["D_l"]) * 1e3, width=0.4, color=col, alpha=0.25, lw=0)
        a.plot(x, p["s_l"], color=col, lw=0.9, marker="o", ms=1.6, label=f"$s_l$, S={float(S_):.2f}")
        bar_handles.append((Patch(color=col, alpha=0.25, lw=0), f"$D_l$ (M={p['M']})"))
        a.axhline(float(S_), color=col, ls=":", lw=0.6)
    for v in (0.10, 0.85):
        a.axhline(v, color=R.OI["grey"], ls=":", lw=0.6)
    a.set_zorder(ax2.get_zorder() + 1)
    a.patch.set_visible(False)
    a.set_xlabel("block")
    a.set_ylabel("block sparsity $s_l$")
    ax2.set_ylabel("OWL outlier ratio $D_l$ ($\\times10^{-3}$, bars)")
    a.set_ylim(0.0, 1.22)
    ax2.set_ylim(0, 75)
    a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    lh, ll = a.get_legend_handles_labels()
    a.legend(lh + [h_ for h_, _ in bar_handles], ll + [l_ for _, l_ in bar_handles], loc="upper center", ncol=2,
             fontsize=5.5, handlelength=1.0, columnspacing=0.6)
    units = qp["units"]
    sens = np.array(qp["sensitivity_dnll"])  # [196, 4] bits (2,3,4,8)
    j3 = qp["bits_options"].index(3)
    M = np.zeros((7, L))
    for u, row in zip(units, sens):
        li, pn = u.split(".", 1)
        M[PROJ.index(pn), int(li[1:])] = row[j3]
    im = b.imshow(np.clip(M, 1e-5, None), aspect="auto", cmap="viridis", norm=LogNorm(vmin=1e-5, vmax=M.max()),
                  interpolation="nearest")
    b.set_yticks(range(7), PSHORT)
    b.set_xlabel("block")
    cb = f.colorbar(im, ax=b, pad=0.01, fraction=0.06, extend="min")
    cb.set_label("dev $\\Delta$NLL at 3 bits (nats)", fontsize=R.SMALL)
    cb.ax.tick_params(labelsize=5.5)
    cmap = ListedColormap([R.OI["verm"], R.OI["orange"], R.OI["sky"], R.OI["blue"]])
    norm = BoundaryNorm([0, 1, 2, 3, 4], 4)
    idx = {2: 0, 3: 1, 4: 2, 8: 3}
    for ax, name in ((c1, "B4"), (c2, "B3")):
        A = np.zeros((7, L))
        for e in qp["allocations"][name]["bits_map"]:
            A[PROJ.index(e["proj"]), e["layer"]] = idx[e["bits"]] + 0.5
        ax.imshow(A, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_yticks(range(7), PSHORT, fontsize=5)
        ax.text(1.01, 0.5, name, transform=ax.transAxes, rotation=90, va="center", fontsize=R.SMALL)
    c1.set_xticks([])
    c2.set_xlabel("block")
    c2.legend([Patch(color=cmap(i)) for i in range(4)], ["2", "3", "4", "8 bit"], ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.38), fontsize=5.5, handlelength=0.8, columnspacing=0.6)
    R.panel(a, "a")
    R.panel(b, "b")
    R.panel(c1, "c")
    R.record(R.save(f, 1, "fig2_allocation.pdf", w, h), S,
             ["results/task1/diagnostics.json", "results/task1/q3proj_allocation.json"])


def figA_q3block():
    w, h = 5.5, 1.35
    qa = R.load("results/task1/q3_allocation.json")
    dn = np.array(qa["probe_dnll"])  # [28, 4]
    f = R.fig(w, h)
    a = f.subplots(1, 1)
    im = a.imshow(np.clip(dn.T, 1e-5, None), aspect="auto", cmap="viridis", norm=LogNorm(vmin=1e-5, vmax=dn.max()),
                  interpolation="nearest")
    a.set_yticks(range(4), [f"{b_} bit" for b_ in qa["bits_options"]])
    a.set_xlabel("block")
    cb = f.colorbar(im, ax=a, pad=0.01, fraction=0.04, extend="min" if (dn <= 1e-5).any() else "neither")
    cb.set_label("isolated dev $\\Delta$NLL", fontsize=R.SMALL)
    b4, b3 = (sorted(set(qa["allocations"][n]["bits"])) for n in ("B4", "B3"))
    assert len(b4) == 1 and len(b3) == 1
    a.text(0.995, 0.97, f"knapsack choice: all blocks {b4[0]}-bit (B4), {b3[0]}-bit (B3)", transform=a.transAxes,
           ha="right", va="top", fontsize=5.5, color="black",
           bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    R.record(R.save(f, 1, "figA_t1_q3block.pdf", w, h), S, ["results/task1/q3_allocation.json"])


# ---------------------------------------------------------------- tables
ROWS = [("FP16", "baseline"), "MID", ("P1 25\\%", "p1_mag_s25"), ("P1 50\\%", "p1_mag_s50"), ("P1 75\\%", "p1_mag_s75"),
        ("P2 25\\%", "p2_wanda_s25"), ("P2 50\\%", "p2_wanda_s50"), ("P2 75\\%", "p2_wanda_s75"),
        ("P3 50\\%", "p3_owl_s50"), ("P3 75\\%", "p3_owl_s75"), ("2:4", "p2_wanda_2of4"), "MID",
        ("Q1 INT8", "q1_int8"), ("Q1 INT4", "q1_int4"), ("Q1 INT3", "q1_int3"), ("Q2 INT8", "q2_awq_int8"),
        ("Q2 INT4", "q2_awq_int4"), ("Q2 INT3", "q2_awq_int3"), ("Q3-block B4 (= Q2)", "q3_mixed_b4"),
        ("Q3-block B3 (= Q2)", "q3_mixed_b3"), ("Q3-proj B4", "q3proj_b4"), ("Q3-proj B3", "q3proj_b3")]


def compress_desc(c):
    st = C[c]["storage"]
    if c == "baseline":
        return "16 b"
    if st["quant_total_bytes"]:
        return f"{st['nominal_bits']} b" if isinstance(st["nominal_bits"], int) else f"{st['nominal_bits']:.0f} b (mixed)"
    return f"{100 * st['mask_pruned_fraction']:.0f}\\%"


def ppl_fmt(v):
    """perplexities >= 10,000 in scientific notation (e.g. 6.39e4), otherwise 2 decimals."""
    return R.num(v, "{:.2f}") if v < 1e4 else f"{v:.2e}".replace("e+0", "e").replace("e+", "e")


def tab1_summary():
    rows = []
    for r in ROWS:
        if r == "MID":
            rows.append("MID")
            continue
        lab, c = r
        x = C[c]
        hs = x["hellaswag"]
        rows.append([lab, compress_desc(c), f"{total_gb(c):.3f}", ppl_fmt(x["wikitext2"]["ppl"]), ppl_fmt(x["c4"]["ppl"]),
                     f"{hs['first_512']['acc_norm']:.3f}", f"{hs['all']['acc_norm']:.3f}" if x["hellaswag_n"] >= 2000 else "",
                     "" if x["output_error_param_weighted"] is None else
                     (f"{x['output_error_param_weighted']:.4f}" if x["output_error_param_weighted"] >= 1e-3 else
                      f"{x['output_error_param_weighted']:.1e}".replace("e-0", "e-"))])
    p = R.write_tab(1, "tab1_summary.tex", "lrrrrrrr",
                    ["Condition", "Sparsity/bits", "GB", "WT2 PPL", "C4 PPL", "HS@512", "HS@2000", "Out. err."], rows)
    R.record(p, S, [SUM])


def tab1_contrasts():
    spec = [("P2$-$P1 @50\\%", "p2_wanda_s50-p1_mag_s50"), ("P3$-$P2 @50\\%", "p3_owl_s50-p2_wanda_s50"),
            ("P3$-$P2 @75\\%", "p3_owl_s75-p2_wanda_s75"), ("Q2$-$Q1 @INT4", "q2_awq_int4-q1_int4"),
            ("Q2$-$Q1 @INT3", "q2_awq_int3-q1_int3"), ("Q3$-$Q2 @B4 (Q3-proj)", "q3proj_b4-q2_awq_int4"),
            ("Q3$-$Q2 @B3 (Q3-proj)", "q3proj_b3-q2_awq_int3")]
    rows = []
    for lab, k in spec:
        x = CT[k]
        a, b = k.split("-")
        n = 2000 if C[a]["hellaswag_n"] >= 2000 and C[b]["hellaswag_n"] >= 2000 else 512
        big = abs(x["wikitext2"]["ppl"]["point"]) > 1000
        fmt = "{:.0f}" if big else "{:.2f}"
        rows.append([lab, R.ci(x["wikitext2"]["ppl"], fmt=fmt), R.ci(x["c4"]["ppl"], fmt=fmt),
                     R.ci(x[f"hs_acc_norm@{n}"], 100, "{:.1f}") + f" ({n})"])
    p = R.write_tab(1, "tab1_contrasts.tex", "lrrr",
                    ["Contrast", "$\\Delta$PPL WT2 [95\\% CI]", "$\\Delta$PPL C4 [95\\% CI]",
                     "$\\Delta$acc\\_norm, pp (subset)"], rows)
    R.record(p, S, [SUM])


def tab1_resources():
    """calib. s = activation-statistics capture; search+apply s = scoring / AWQ search + apply (+ pack/verify).
    Times marked double-dagger are paid once and shared across rows (FP16 activation statistics for pruning; one AWQ
    capture + search reused for INT8/INT4/INT3), not per row."""
    global PRUNE_STATS_S
    PRUNE_STATS_S = R.load("results/task1/diagnostics.json")["selection_effort_gpu_seconds"]["calibration_stats_baseline_s"]
    spec = [("FP16", "baseline"), ("Unstruct. 50\\% (P2)", "p2_wanda_s50"), ("2:4 50\\%", "p2_wanda_2of4"),
            ("INT8 (Q2)", "q2_awq_int8"), ("INT4 (Q2)", "q2_awq_int4"), ("INT3 (Q2)", "q2_awq_int3")]
    dd = "$^\\ddagger$"
    rows = []
    for lab, c in spec:
        x = C[c]
        cs = R.load(f"results/task1/{c}/summary.json").get("condition_info", {}).get("compress_seconds", {})
        nb = 16 if c == "baseline" else (x["storage"]["nominal_bits"] if x["storage"]["quant_total_bytes"] else 16)
        r = x["latency_drift_adjusted_ratio"]
        if c == "baseline":
            calib, sa = "", ""
        elif c.startswith("p2"):
            calib = f"{PRUNE_STATS_S:.0f}{dd} + {cs['stat_load_s']:.1f}"
            sa = f"{cs['score_and_mask_s'] + cs['search_s'] + cs['apply_s']:.1f}"
        else:
            calib = f"{cs['stat_capture_s_shared_all_bits']:.0f}{dd}"
            sa = f"{cs['search_s_shared_all_bits']:.0f}{dd} + {cs['apply_and_pack_verify_s']:.0f}"
        rows.append([lab, str(nb), f"{total_gb(c):.3f}", f"{max(x['peak_alloc_bytes'].values()) / GB:.3f}",
                     f"{r['prefill_512']:.2f}", f"{r['prefill_2048']:.2f}", f"{r['decode_64_after_512']:.2f}", calib, sa])
    q = R.load("results/task1/q2_awq_int4/summary.json")["condition_info"]["compress_seconds"]
    shared = q["stat_capture_s_shared_all_bits"] + q["search_s_shared_all_bits"]
    p = R.write_tab(1, "tab1_resources.tex", "lrrrrrrrr",
                    ["Condition", "bits/w", "Packed GB", "Peak GB$^\\dagger$", "P512", "P2048", "D64",
                     "calib. s", "search+apply s"], rows,
                    "% dagger: all conditions execute dense FP16 kernels (zeros / dequantized weights); ratios are "
                    "drift-adjusted vs interleaved FP16 references\n"
                    f"% double dagger: paid once and shared across rows, not per row: the FP16 activation-statistics "
                    f"capture ({PRUNE_STATS_S:.1f} s) is loaded by every pruning condition; the Q2 AWQ capture + search "
                    f"({shared:,.0f} s) is run once and reused for INT8/INT4/INT3 (only apply+pack/verify is per row)")
    R.record(p, S, [SUM, "results/task1/diagnostics.json"] + [f"results/task1/{c}/summary.json" for _, c in spec])


QCFG = [("FP16", "baseline"), ("P1@50", "p1_mag_s50"), ("P2@50", "p2_wanda_s50"), ("P3@50", "p3_owl_s50"),
        ("Q2-INT4", "q2_awq_int4"), ("Q3-proj-B4", "q3proj_b4"), ("Q2-INT3", "q2_awq_int3")]


def _hs_scores(c, order):
    from atml4.common.io import load_jsonl_result
    rows = {r["order"]: r for r in load_jsonl_result(R.ROOT / f"results/task1/{c}/hellaswag.jsonl")[1]}
    r = rows[order]
    return [ll / n for ll, n in zip(r["loglik"], r["char_len"])], r["pred_acc_norm"]


def _trunc(s, n):
    w = s.split()
    return " ".join(w[:n]) + (" \\ldots" if len(w) > n else "")


def _esc(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("$", "\\$").replace("#", "\\#")


def tab_qual(name, cases_sel):
    """Per-case header row spans all columns in a fixed-width p{} cell; the model's choice (highest per-character
    log-likelihood = pred_acc_norm) is bold; checkmark when it equals gold. The gold ending is marked in the column
    header when every case shown shares it, otherwise only in the case header."""
    q = R.load("results/task1/qualitative.json")
    cases = [c for c in q["cases"] if c["order"] in cases_sel]
    golds = {c["gold"] for c in cases}
    rows = []
    for case in cases:
        g = case["gold"]
        rows.append([f"\\multicolumn{{6}}{{@{{}}p{{3.2in}}@{{}}}}{{HellaSwag ID {_esc(case['id'])} (first-512 order "
                     f"{case['order']}): {_esc(_trunc(case['context'], 15))} Gold [{g}]: "
                     f"{_esc(_trunc(case['endings'][g], 10))}}}"])
        for lab, c in QCFG:
            sc, pred = _hs_scores(c, case["order"])
            assert pred == int(np.argmax(sc)), (c, case["order"])
            cells = [f"{v:.3f}".replace("-", "$-$") for v in sc]
            cells[pred] = "\\textbf{\\boldmath " + cells[pred] + "}"
            rows.append([lab, str(pred) + (" \\checkmark" if pred == g else "")] + cells)
        rows.append("MID")
    if rows and rows[-1] == "MID":
        rows.pop()
    ends = [f"end {k}" + (" (gold)" if golds == {k} else "") for k in range(4)]
    p = R.write_tab(1, name, "llrrrr", ["Model", "Chosen", *ends], rows,
                    "% per-character log-likelihood of each ending; bold = the model's choice (highest score); "
                    "checkmark = choice equals gold")
    R.record(p, S, ["results/task1/qualitative.json"] + [f"results/task1/{c}/hellaswag.jsonl" for _, c in QCFG])


def tabA_full():
    rows = []
    for r in ROWS:
        if r == "MID":
            rows.append("MID")
            continue
        lab, c = r
        if c == "baseline":
            continue
        x = C[c]
        vb = x["hellaswag"].get("vs_baseline", {})
        big = x["wikitext2"]["ppl"] > 1000
        fmt = "{:.0f}" if big else "{:.2f}"
        rows.append([lab, R.ci(x["wikitext2"]["vs_baseline"]["ppl"], fmt=fmt), R.ci(x["c4"]["vs_baseline"]["ppl"], fmt=fmt),
                     R.ci(vb.get("acc_norm@512"), 100, "{:.1f}"),
                     R.ci(vb.get("acc_norm@2000"), 100, "{:.1f}") if x["hellaswag_n"] >= 2000 else ""])
    p = R.write_tab(1, "tabA_t1_full.tex", "lrrrr",
                    ["Condition", "$\\Delta$PPL WT2", "$\\Delta$PPL C4", "$\\Delta$acc\\_n@512 (pp)",
                     "$\\Delta$acc\\_n@2000 (pp)"], rows, "% all vs FP16; paired bootstrap 95% CI, 2,000 resamples")
    R.record(p, S, [SUM])


def tabA_p3grid():
    g = R.load("results/task1/diagnostics.json")["p3_dev_grid_compact"]
    sel = R.load("results/task1/p3_selection.json")["selected"]
    Ms, ls = [3, 5, 7, 10], [0.02, 0.05, 0.08, 0.12, 0.16, 0.2]
    rows = []
    for S_ in ("0.5", "0.75"):
        rows.append([f"\\multicolumn{{7}}{{l}}{{\\textit{{S = {S_}; uniform (P2) dev PPL "
                     f"{g[S_]['MNone_l0.0']:.2f}}}}}"])
        for M in Ms:
            cells = []
            for lam in ls:
                v = g[S_][f"M{M}_l{lam}"]
                s = f"{v:.2f}" if v < 1000 else f"{v:.0f}"
                if sel[S_]["M"] == M and abs(sel[S_]["lambda"] - lam) < 1e-9:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            rows.append([f"M={M}"] + cells)
        if S_ == "0.5":
            rows.append("MID")
    p = R.write_tab(1, "tabA_t1_p3grid.tex", "lrrrrrr", ["", *[f"$\\lambda$={v}" for v in ls]], rows,
                    "% C4 development PPL (32 x 512 tokens); selected in bold")
    R.record(p, S, ["results/task1/diagnostics.json", "results/task1/p3_selection.json"])


def tabA_retiming():
    rt = R.load("results/task1/retiming.json")
    rq = R.load("results/task1/retiming_q3proj.json")
    rows = []
    for src, data in (("full", rt), ("Q3-proj pass", rq)):
        for c, v in data["per_condition"].items():
            if src == "Q3-proj pass" and c == "baseline":
                continue
            rows.append([c.replace("_", "\\_"), f"{v['prefill_512']['median_all_reps_ms']:.1f}",
                         f"{v['prefill_512']['drift_adjusted_ratio']:.3f}", f"{v['prefill_2048']['median_all_reps_ms']:.1f}",
                         f"{v['prefill_2048']['drift_adjusted_ratio']:.3f}",
                         f"{v['decode_64_after_512']['median_all_reps_ms']:.0f}",
                         f"{v['decode_64_after_512']['drift_adjusted_ratio']:.3f}", src])
    cs = rt["clock_summary"]
    note = (f"% clocks (full pass): SM {cs['sm_mhz']['min']:.0f}-{cs['sm_mhz']['max']:.0f} MHz (median "
            f"{cs['sm_mhz']['median']:.0f}), {cs['temp_c']['min']:.0f}-{cs['temp_c']['max']:.0f} C, "
            f"{cs['power_w']['min']:.0f}-{cs['power_w']['max']:.0f} W; clock lock not permitted")
    p = R.write_tab(1, "tabA_t1_retiming.tex", "lrrrrrrl",
                    ["Condition", "P512 ms", "ratio", "P2048 ms", "ratio", "D64 ms", "ratio", "pass"], rows, note)
    R.record(p, S, ["results/task1/retiming.json", "results/task1/retiming_q3proj.json"])


def tabA_pruning_diag():
    ps = R.load("results/task1/pruning_stats.json")["per_projection"]
    rows = []
    for p_, s in zip(PROJ, PSHORT):
        xs = [x for x in ps if x["name"] == p_]
        rows.append([s, f"{np.mean([x['spearman_absW_wanda'] for x in xs]):.3f}",
                     *[f"{np.mean([x['jaccard_pruned_mag_vs_wanda'][k] for x in xs]):.3f}" for k in ("0.25", "0.5", "0.75")],
                     *[f"{1e3 * np.mean([x['owl_outlier_ratio'][k] for x in xs]):.2f}" for k in ("3", "5", "7", "10")]])
    p = R.write_tab(1, "tabA_t1_pruning_diag.tex", "lrrrrrrrr",
                    ["Proj.", "Spearman", "J@25", "J@50", "J@75", "$D$ M=3", "M=5", "M=7", "M=10"], rows,
                    "% mean over 28 blocks; Spearman(|W|, Wanda score) on 200k sampled weights; Jaccard of pruned "
                    "sets; OWL outlier ratio x 1e-3")
    R.record(p, S, ["results/task1/pruning_stats.json"])


def main():
    R.setup()
    fig1()
    fig2()
    figA_q3block()
    tab1_summary()
    tab1_contrasts()
    tab1_resources()
    tab_qual("tab1_qualitative.tex", {0})
    q = R.load("results/task1/qualitative.json")
    tab_qual("tabA_t1_qualitative_all.tex", {c["order"] for c in q["cases"]})
    tabA_full()
    tabA_p3grid()
    tabA_retiming()
    tabA_pruning_diag()
    R.dump_manifest(S)
    print("task1 outputs:", len(R.MANIFEST))


if __name__ == "__main__":
    main()
