# Report figure/table manifest

| output | script | source files |
|---|---|---|
| `report/figs/task1/fig1_tradeoffs.pdf` | `scripts/report/task1.py` | `results/task1/summary_all.json`, `results/task1/pruning_stats.json` |
| `report/figs/task1/fig2_allocation.pdf` | `scripts/report/task1.py` | `results/task1/diagnostics.json`, `results/task1/q3proj_allocation.json` |
| `report/figs/task1/figA_t1_q3block.pdf` | `scripts/report/task1.py` | `results/task1/q3_allocation.json` |
| `report/figs/task1/tab1_contrasts.tex` | `scripts/report/task1.py` | `results/task1/summary_all.json` |
| `report/figs/task1/tab1_qualitative.tex` | `scripts/report/task1.py` | `results/task1/qualitative.json`, `results/task1/baseline/hellaswag.jsonl`, `results/task1/p1_mag_s50/hellaswag.jsonl`, `results/task1/p2_wanda_s50/hellaswag.jsonl`, `results/task1/p3_owl_s50/hellaswag.jsonl`, `results/task1/q2_awq_int4/hellaswag.jsonl`, `results/task1/q3proj_b4/hellaswag.jsonl`, `results/task1/q2_awq_int3/hellaswag.jsonl` |
| `report/figs/task1/tab1_resources.tex` | `scripts/report/task1.py` | `results/task1/summary_all.json`, `results/task1/baseline/summary.json`, `results/task1/p2_wanda_s50/summary.json`, `results/task1/p2_wanda_2of4/summary.json`, `results/task1/q2_awq_int8/summary.json`, `results/task1/q2_awq_int4/summary.json`, `results/task1/q2_awq_int3/summary.json` |
| `report/figs/task1/tab1_summary.tex` | `scripts/report/task1.py` | `results/task1/summary_all.json` |
| `report/figs/task1/tabA_t1_full.tex` | `scripts/report/task1.py` | `results/task1/summary_all.json` |
| `report/figs/task1/tabA_t1_p3grid.tex` | `scripts/report/task1.py` | `results/task1/diagnostics.json`, `results/task1/p3_selection.json` |
| `report/figs/task1/tabA_t1_pruning_diag.tex` | `scripts/report/task1.py` | `results/task1/pruning_stats.json` |
| `report/figs/task1/tabA_t1_qualitative_all.tex` | `scripts/report/task1.py` | `results/task1/qualitative.json`, `results/task1/baseline/hellaswag.jsonl`, `results/task1/p1_mag_s50/hellaswag.jsonl`, `results/task1/p2_wanda_s50/hellaswag.jsonl`, `results/task1/p3_owl_s50/hellaswag.jsonl`, `results/task1/q2_awq_int4/hellaswag.jsonl`, `results/task1/q3proj_b4/hellaswag.jsonl`, `results/task1/q2_awq_int3/hellaswag.jsonl` |
| `report/figs/task1/tabA_t1_retiming.tex` | `scripts/report/task1.py` | `results/task1/retiming.json`, `results/task1/retiming_q3proj.json` |
| `report/figs/task2/fig3_attention_ops.pdf` | `scripts/report/task2.py` | `results/task2/partA/timing.jsonl`, `results/task2/partA/correctness.json` |
| `report/figs/task2/fig4_mqar.pdf` | `scripts/report/task2.py` | `results/task2/partB/summary.json`, `results/task2/partB/runs/dense_K16_mixed_s6304.json`, `results/task2/partB/runs/dense_K16_mixed_s6305.json`, `results/task2/partB/runs/dense_K16_mixed_s6306.json` |
| `report/figs/task2/fig5_kv.pdf` | `scripts/report/task2.py` | `results/task2/partC/summary.json`, `results/task2/partC/quality_s0.jsonl`, `results/task2/partC/quality_s1.jsonl` |
| `report/figs/task2/figA_t2_curriculum.pdf` | `scripts/report/task2.py` | `results/task2/partB/summary.json`, `results/task2/partB/runs/window_K16_curriculum_s6304.json`, `results/task2/partB/runs/window_K16_curriculum_s6305.json`, `results/task2/partB/runs/window_K16_curriculum_s6306.json` |
| `report/figs/task2/tab2_backends.tex` | `scripts/report/task2.py` | `results/task2/partA/backend_table.json` |
| `report/figs/task2/tab2_kv.tex` | `scripts/report/task2.py` | `results/task2/partC/summary.json` |
| `report/figs/task2/tab2_kv_contrasts.tex` | `scripts/report/task2.py` | `results/task2/partC/summary.json` |
| `report/figs/task2/tab2_mqar_resources.tex` | `scripts/report/task2.py` | `results/task2/partB/train_step_timing.json`, `results/task2/partB/inference_sweep.jsonl` |
| `report/figs/task2/tabA_t2_dilated.tex` | `scripts/report/task2.py` | `results/task2/partB/dilated_alignment.json` |
| `report/figs/task2/tabA_t2_kquant.tex` | `scripts/report/task2.py` | `results/task2/partC/k_quant_diagnostics.json`, `results/task2/partC/k_quant_convention_compare.json` |
| `report/figs/task2/tabA_t2_kv_breakdown.tex` | `scripts/report/task2.py` | `results/task2/partC/quality_s0.jsonl`, `results/task2/partC/quality_s1.jsonl` |
| `report/figs/task2/tabA_t2_kv_generations.tex` | `scripts/report/task2.py` | `results/task2/partC/summary.json` |
| `report/figs/task2/tabA_t2_mqar_bins.tex` | `scripts/report/task2.py` | `results/task2/partB/summary.json` |
| `report/figs/task2/tabA_t2_mqar_full.tex` | `scripts/report/task2.py` | `results/task2/partB/summary.json` |
| `report/figs/task2/tabA_t2_packed.tex` | `scripts/report/task2.py` | `results/task2/partC/summary.json` |
| `report/figs/task2/tabA_t2_partA_full.tex` | `scripts/report/task2.py` | `results/task2/partA/timing.jsonl`, `results/task2/partA/correctness.json` |
| `report/figs/task3/fig6_distill_align.pdf` | `scripts/report/task3.py` | `results/task3/summary.json`, `results/task3/align_rows.jsonl.gz`, `results/task3/distill_off_s6304.json`, `results/task3/distill_off_s6305.json`, `results/task3/distill_on_s6304.json`, `results/task3/distill_on_s6305.json` |
| `report/figs/task3/fig7_spec_cost.pdf` | `scripts/report/task3.py` | `results/task3/summary.json`, `results/task3/step6_g2.jsonl`, `results/task3/step6_g4.jsonl` |
| `report/figs/task3/tab3_costs.tex` | `scripts/report/task3.py` | `results/task3/summary.json`, `results/task3/teacher_collection.jsonl.gz` |
| `report/figs/task3/tab3_sampler.tex` | `scripts/report/task3.py` | `results/task3/toy_chisquare.json` |
| `report/figs/task3/tab3_spec.tex` | `scripts/report/task3.py` | `results/task3/summary.json` |
| `report/figs/task3/tabA_t3_accuracy.tex` | `scripts/report/task3.py` | `results/task3/summary.json` |
| `report/figs/task3/tabA_t3_align_bins.tex` | `scripts/report/task3.py` | `results/task3/align_rows.jsonl.gz` |
| `report/figs/task3/tabA_t3_align_domain.tex` | `scripts/report/task3.py` | `results/task3/align_rows.jsonl.gz` |
| `report/figs/task3/tabA_t3_baseline.tex` | `scripts/report/task3.py` | `results/task3/summary.json` |
| `report/figs/task3/tabA_t3_chisq_joint.tex` | `scripts/report/task3.py` | `results/task3/toy_chisquare.json` |
| `report/figs/task3/tabA_t3_disagree.tex` | `scripts/report/task3.py` | `results/task3/summary.json` |
| `report/figs/task3/tabA_t3_fixed64.tex` | `scripts/report/task3.py` | `results/task3/summary.json` |
| `report/figs/task3/tabA_t3_merge.tex` | `scripts/report/task3.py` | `results/task3/summary.json` |
