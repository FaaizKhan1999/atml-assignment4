# GPU ledger

One row per Kaggle kernel run (appended by `scripts/kaggle_run.py fetch`). Wall time is the bootstrap's own wall clock (excludes Kaggle queue/startup). GPU-hours = wall hours for GPU kernels, 0 for CPU kernels.

**Running total GPU-hours: 6.661**

| date (UTC) | kernel | accelerator (nvidia-smi) | wall time (h) | GPU-hours | status | commit | purpose |
|---|---|---|---|---|---|---|---|
| 2026-11-25T16:15:27+00:00 | focusedkognition/atml4-data-prep | none (CPU) | 0.020 | 0.000 | ok | b3c8016772 | Task data prep: setup_data.py --public + check_setup.py |
| 2026-11-25T16:15Z | focusedkognition/atml4-probe-t4x2 | never started | 0.000 | 0.000 | QUEUED ~44 min, deleted (CLI has no cancel) | b3c8016772 | accelerator/concurrency probe (machine_shape=NvidiaTeslaT4) |
| 2026-11-25T16:15Z | focusedkognition/atml4-probe-default | never started | 0.000 | 0.000 | QUEUED ~44 min, deleted (CLI has no cancel) | b3c8016772 | accelerator/concurrency probe (enable_gpu, no machine_shape) |
| 2026-11-25T17:00:14+00:00 | focusedkognition/atml4-gpu-smoke | 2x Tesla T4 | 0.039 | 0.039 | ok; prefix-bank smoke FAILED rc=1 (course setup_data.py: generate() rejects generator=) | 40daf27001 | GPU smoke: env lock, meta, 0.5B FP16 fwd, SDPA probe, prefix-bank smoke (2,4 prompts) |
| 2026-11-25T17:57:41+00:00 | focusedkognition/atml4-prefix-banks | 2x Tesla T4 | 0.268 | 0.268 | ok | 214c0ac5a3 | Task3 prefix banks: smoke(2) -> estimate -> full build if <=2.5h |
| 2026-11-25T18:09:12+00:00 | focusedkognition/atml4-t1-baseline | 2x Tesla T4 | 0.203 | 0.203 | baseline ok; lm-eval crosscheck DISAGREES (rc=2) | 8b6cba3705 | Task1 Step1: FP16 baseline (calib stats, PPL x2, HellaSwag 2000, timing) + lm-eval crosscheck(100) |
| 2026-11-25T18:47:21+00:00 | focusedkognition/atml4-t1-fixed | 2x Tesla T4 | 1.377 | 1.377 | ok | bac75d66ea | Task1 fixed: lm-eval sidecheck; P1/P2/2:4 (cuda:0) || Q1/Q2 (cuda:1); exclusive timing |
| 2026-11-25T21:34:01+00:00 | focusedkognition/atml4-t1-final | 2x Tesla T4 | 1.057 | 1.057 | ok | e5193e7b86 | Task1 final: P3 grid+evals, Q3 probes+knapsack+evals, Q2 outerr fix || ; exclusive ABBA retiming |
| 2026-11-25T21:44:23+00:00 | focusedkognition/atml4-t2-ab | 2x Tesla T4 | 0.937 | 0.937 | ok | d651facfb1 | Task2 A (exclusive) + B1/B2 39 MQAR runs + exclusive sweep |
| 2026-11-26T04:55:22+00:00 | focusedkognition/atml4-t2c-validate | 2x Tesla T4 | 0.185 | 0.185 | ok | 3ee16ce8fa | Task2C validation: C1 on real model, FP16 accuracy 180, bytes by serialization, all policies on 3 prompts |
| 2026-11-26T05:26:14+00:00 | focusedkognition/atml4-t2c-kdiag | 2x Tesla T4 | 0.026 | 0.026 | ok | 6621e46312 | Task2C K-quant diagnostics a-e on 2 prompts (save real K/V) |
| 2026-11-26T04:40:56+00:00 | focusedkognition/atml4-q3proj-b3 | 2x Tesla T4 | 0.937 | 0.937 | ok | 9f06b3212c | Task1 Q3-proj probes+MILP+evals (cuda0/cuda1 split) || Task2 B3 curriculum x3; exclusive short ABBA retiming |
| 2026-11-26T05:37:18+00:00 | focusedkognition/atml4-t2c-revalidate | 2x Tesla T4 | 0.199 | 0.199 | ok | 8a766451a4 | Task2C re-validation after KV quantizer fix |
| 2026-11-26T05:40:11+00:00 | focusedkognition/atml4-t3-a | 2x Tesla T4 | 0.793 | 0.793 | ok | 0ad859a28c | Task3 T3-a: real-model sampler checks, bank (iii), teacher collection || ; exclusive Step 2 baseline |
| 2026-11-26T06:23:18+00:00 | focusedkognition/atml4-t2c-full | 2x Tesla T4 | 0.612 | 0.612 | ok | af7432327f | Task2C full run: quality 180 x (6 required + 2 supplementary zp) on 2 GPUs; exclusive timing subset; packed benchmark |
| 2026-11-26T06:42:33+00:00 | focusedkognition/atml4-t3-b | 2x Tesla T4 | 0.028 | 0.028 | FAILED(rc=1,err=None) | 8c0129beac | Task3 T3-b: 4 distillation runs (2/GPU), merge+check, own banks, Step 5 alignment, standalone accuracy |
