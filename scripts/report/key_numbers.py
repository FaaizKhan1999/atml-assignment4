#!/usr/bin/env python3
"""Write report/figs/NUMBERS.md: 3-6 key numbers per figure/table, computed from results/ (for captions)."""
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import rstyle as R  # noqa: E402


def t1():
    S = R.load("results/task1/summary_all.json")
    C, CT = S["conditions"], S["contrasts"]
    g = R.load("results/task1/diagnostics.json")
    ps = R.load("results/task1/pruning_stats.json")["per_projection"]
    qp = R.load("results/task1/q3proj_allocation.json")
    rt = R.load("results/task1/retiming.json")
    out = ["## Task 1", "",
           "### fig1_tradeoffs.pdf",
           f"- FP16: {C['baseline']['storage']['dense_fp16_total_bytes'] / 1e9:.3f} GB, WT2 PPL {C['baseline']['wikitext2']['ppl']:.2f}, "
           f"HS acc_norm@512 {C['baseline']['hellaswag']['first_512']['acc_norm']:.3f}",
           f"- 50% sparsity (sparse16 {C['p2_wanda_s50']['storage']['sparse16_total_bytes'] / 1e9:.3f} GB): WT2 PPL P1 {C['p1_mag_s50']['wikitext2']['ppl']:.2f}, "
           f"P2 {C['p2_wanda_s50']['wikitext2']['ppl']:.2f}, P3 {C['p3_owl_s50']['wikitext2']['ppl']:.2f}, 2:4 {C['p2_wanda_2of4']['wikitext2']['ppl']:.2f}",
           f"- INT4 ({C['q2_awq_int4']['storage']['quant_total_bytes'] / 1e9:.3f} GB): WT2 PPL Q1 {C['q1_int4']['wikitext2']['ppl']:.2f}, "
           f"Q2 {C['q2_awq_int4']['wikitext2']['ppl']:.2f}, Q3-proj B4 {C['q3proj_b4']['wikitext2']['ppl']:.2f}; INT3 Q1 {C['q1_int3']['wikitext2']['ppl']:.2f} / Q2 {C['q2_awq_int3']['wikitext2']['ppl']:.2f}",
           f"- 75%: P1 {C['p1_mag_s75']['wikitext2']['ppl']:.3g}, P2 {C['p2_wanda_s75']['wikitext2']['ppl']:.3g}, P3 {C['p3_owl_s75']['wikitext2']['ppl']:.2f}",
           f"- Jaccard(magnitude, Wanda) mean over 196 matrices at 25/50/75%: " + ", ".join(
               f"{np.mean([x['jaccard_pruned_mag_vs_wanda'][k] for x in ps]):.3f}" for k in ("0.25", "0.5", "0.75")),
           "", "### fig2_allocation.pdf",
           f"- P3 S=0.50: M={g['p3_blocks']['0.5']['M']}, lambda={g['p3_blocks']['0.5']['lambda']}, s_l in "
           f"[{min(g['p3_blocks']['0.5']['s_l']):.3f}, {max(g['p3_blocks']['0.5']['s_l']):.3f}]; S=0.75: M={g['p3_blocks']['0.75']['M']}, "
           f"lambda={g['p3_blocks']['0.75']['lambda']}, s_l in [{min(g['p3_blocks']['0.75']['s_l']):.3f}, {max(g['p3_blocks']['0.75']['s_l']):.3f}] "
           f"({sum(abs(v - 0.85) < 1e-9 for v in g['p3_blocks']['0.75']['s_l'])} blocks at the 0.85 cap)",
           f"- Spearman(D_l, P2 block output error): {g['p3_blocks']['0.5']['spearman_D_vs_P2_err']['rho']:.2f} (S=0.5), "
           f"{g['p3_blocks']['0.75']['spearman_D_vs_P2_err']['rho']:.2f} (S=0.75)",
           f"- Q3-proj bit counts B4: " + str({n: {b: c for b, c in v.items() if c} for n, v in qp['allocations']['B4']['counts_proj_type_x_bits'].items()}),
           f"- Q3-proj bit counts B3: " + str({n: {b: c for b, c in v.items() if c} for n, v in qp['allocations']['B3']['counts_proj_type_x_bits'].items()}),
           f"- B4/B3 realised bytes {qp['allocations']['B4']['total_bytes']:,} / {qp['allocations']['B3']['total_bytes']:,} (slack 6,528 each); "
           f"predicted vs realised dev dNLL B4 {qp['allocations']['B4']['predicted_dev_dnll']:.4f}/{qp['allocations']['B4']['realised_dev_dnll']:.4f}, "
           f"B3 {qp['allocations']['B3']['predicted_dev_dnll']:.4f}/{qp['allocations']['B3']['realised_dev_dnll']:.4f}",
           "", "### tab1_summary.tex / tabA_t1_full.tex",
           f"- HS acc_norm@2000: FP16 {C['baseline']['hellaswag']['all']['acc_norm']:.3f}, P2@50 {C['p2_wanda_s50']['hellaswag']['all']['acc_norm']:.3f}, "
           f"Q2 INT4 {C['q2_awq_int4']['hellaswag']['all']['acc_norm']:.3f}, Q2 INT3 {C['q2_awq_int3']['hellaswag']['all']['acc_norm']:.3f}, Q3-proj B4 {C['q3proj_b4']['hellaswag']['all']['acc_norm']:.3f}",
           "", "### tab1_contrasts.tex"]
    for k in ("p2_wanda_s50-p1_mag_s50", "p3_owl_s50-p2_wanda_s50", "q2_awq_int4-q1_int4", "q3proj_b4-q2_awq_int4",
              "q3proj_b3-q2_awq_int3"):
        c = CT[k]["wikitext2"]["ppl"]
        out.append(f"- {k}: dPPL WT2 {c['point']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]")
    out += ["", "### tab1_resources.tex",
            f"- peak allocated {max(C['baseline']['peak_alloc_bytes'].values()) / 1e9:.3f} GB for every condition (dense FP16 execution)",
            f"- prefill-512 drift-adjusted ratio P2@50 {C['p2_wanda_s50']['latency_drift_adjusted_ratio']['prefill_512']:.2f}, "
            f"Q2 INT4 {C['q2_awq_int4']['latency_drift_adjusted_ratio']['prefill_512']:.2f}; decode-64 within "
            f"{min(v['latency_drift_adjusted_ratio']['decode_64_after_512'] for v in C.values()):.2f}-"
            f"{max(v['latency_drift_adjusted_ratio']['decode_64_after_512'] for v in C.values()):.2f}",
            f"- clocks: SM {rt['clock_summary']['sm_mhz']['min']:.0f}-{rt['clock_summary']['sm_mhz']['max']:.0f} MHz, "
            f"{rt['clock_summary']['temp_c']['min']:.0f}-{rt['clock_summary']['temp_c']['max']:.0f} C; lock not permitted", ""]
    return out


def t2():
    tim = R.load("results/task2/partA/timing.jsonl")
    cor = R.load("results/task2/partA/correctness.json")["grid"]
    G = R.load("results/task2/partB/summary.json")["groups"]
    ts = R.load("results/task2/partB/train_step_timing.json")
    P = R.load("results/task2/partC/summary.json")

    def t(op, dt, L, mode="forward_backward"):
        return next(r for r in tim if r["op"] == op and r["dtype"] == dt and r["L"] == L and r["mode"] == mode)
    f16 = [r for r in cor if r["dtype"] == "fp16" and r["status"] == "ok"]
    out = ["## Task 2", "", "### fig3_attention_ops.pdf",
           f"- L=8192 FP16 fwd+bwd: SDPA {t('sdpa_default', 'fp16', 8192)['median']:.1f} ms, dense {t('dense', 'fp16', 8192)['median']:.1f} ms, "
           f"tiled-64 {t('tiled64', 'fp16', 8192)['median']:.0f} ms",
           f"- L=8192 peak allocated fwd+bwd: dense FP32 {t('dense', 'fp32', 8192)['max_memory_allocated_bytes'] / 2 ** 30:.2f} GiB, "
           f"SDPA FP16 {t('sdpa_default', 'fp16', 8192)['max_memory_allocated_bytes'] / 2 ** 20:.0f} MiB",
           f"- FP16 max output error over all L/ops: {min(r['out']['max_abs'] for r in f16):.1e}-{max(r['out']['max_abs'] for r in f16):.1e}; "
           f"gradients up to {max(max(r[k]['max_abs'] for k in ('dq', 'dk', 'dv')) for r in f16):.1e}",
           "- no OOM; FLASH and CUDNN unsupported on T4, default dispatch EFFICIENT", "",
           "### fig4_mqar.pdf / tabA_t2_mqar_full.tex",
           f"- dense K16 mixed: all 9 cells 1.00; with W=64 at test: near/far/mixed "
           f"{G['dense_K16_mixed']['test']['K16_near']['mean']:.2f}/.. (see table)",
           f"- window K16 mixed: near/far/mixed {G['window_K16_mixed']['test']['K16_near']['mean']:.3f}/"
           f"{G['window_K16_mixed']['test']['K16_far']['mean']:.3f}/{G['window_K16_mixed']['test']['K16_mixed']['mean']:.3f}; "
           f"curriculum {G['window_K16_curriculum']['test']['K16_near']['mean']:.4f}/{G['window_K16_curriculum']['test']['K16_far']['mean']:.4f}/"
           f"{G['window_K16_curriculum']['test']['K16_mixed']['mean']:.4f} (pre-registered prediction 0.501)",
           f"- linear K16 mixed {G['linear_K16_mixed']['test']['K16_mixed']['mean']:.3f} +/- {G['linear_K16_mixed']['test']['K16_mixed']['sd']:.3f}; "
           f"dilated K4 {G['dilated_K4_mixed']['test']['K4_mixed']['mean']:.3f}, K16 {G['dilated_K16_mixed']['test']['K16_mixed']['mean']:.3f}",
           f"- curriculum distance bins: " + ", ".join(f"{b} {G['window_K16_curriculum']['dist_bins']['K16_mixed'][b]['mean']:.3f}"
                                                        for b in ("[0,32)", "[32,64)", "[64,128)", "[128,256)")),
           "", "### tab2_mqar_resources.tex",
           f"- train step ms: " + ", ".join(f"{a} {v['median']:.1f}" for a, v in ts["per_arch"].items()),
           f"- pairs/head T=256: dense {ts['theory']['pairs_total_per_head']['dense']}, window {ts['theory']['pairs_total_per_head']['window']}",
           "", "### tab2_kv.tex / fig5_kv.pdf / tab2_kv_contrasts.tex"]
    T = P["table"]
    for m in ("sink_recent", "qaware", "int4_tok", "int4_chan", "hybrid", "int4_tok_zp"):
        out.append(f"- {m}: accuracy {T[m]['accuracy']:.3f}, NLL {T[m]['ref_answer_nll_mean']:.3f}, "
                   f"{100 * T[m]['fraction_of_M16']['mean']:.2f}% of FP16 bytes")
    step = [r for r in P["packed_bench"]["rows"] if r["kind"] == "decode_step" and r["L_tag"] == "actual"]
    out.append("- decode step FP16 vs unpack+step (ms): " + ", ".join(
        f"B{r['B']} L{r['L']} {r['fp16_cache_step_ms']['median']:.1f}/{r['int4_chan_unpack_dequant_plus_step_ms']['median']:.1f}" for r in step))
    out.append(f"- sink+recent correct only when evidence at the end: start/middle/end "
               + "/".join(f"{P['breakdown']['evidence_position'][p]['sink_recent']['accuracy']:.2f}" for p in ("start", "middle", "end")))
    return out + [""]


def main():
    md = ["# Key numbers per figure/table (generated by scripts/report/key_numbers.py from results/)", ""] + t1() + t2()
    tp = R.ROOT / "results/task3/summary.json"
    if (R.ROOT / "results/task3/step6_g2.jsonl").exists():
        md += ["## Task 3", "(see scripts/report/task3.py output)"]
    (R.OUT / "NUMBERS.md").write_text("\n".join(md) + "\n")
    print("NUMBERS.md written", tp.exists())


if __name__ == "__main__":
    main()
