# GPU ledger

One row per Kaggle kernel run (appended by `scripts/kaggle_run.py fetch`). Wall time is the bootstrap's own wall clock (excludes Kaggle queue/startup). GPU-hours = wall hours for GPU kernels, 0 for CPU kernels.

**Running total GPU-hours: 0.039**

| date (UTC) | kernel | accelerator (nvidia-smi) | wall time (h) | GPU-hours | status | commit | purpose |
|---|---|---|---|---|---|---|---|
| 2026-11-25T16:15:27+00:00 | focusedkognition/atml4-data-prep | none (CPU) | 0.020 | 0.000 | ok | b3c8016772 | Task data prep: setup_data.py --public + check_setup.py |
| 2026-11-25T16:15Z | focusedkognition/atml4-probe-t4x2 | never started | 0.000 | 0.000 | QUEUED ~44 min, deleted (CLI has no cancel) | b3c8016772 | accelerator/concurrency probe (machine_shape=NvidiaTeslaT4) |
| 2026-11-25T16:15Z | focusedkognition/atml4-probe-default | never started | 0.000 | 0.000 | QUEUED ~44 min, deleted (CLI has no cancel) | b3c8016772 | accelerator/concurrency probe (enable_gpu, no machine_shape) |
| 2026-11-25T17:00:14+00:00 | focusedkognition/atml4-gpu-smoke | 2x Tesla T4 | 0.039 | 0.039 | ok; prefix-bank smoke FAILED rc=1 (course setup_data.py: generate() rejects generator=) | 40daf27001 | GPU smoke: env lock, meta, 0.5B FP16 fwd, SDPA probe, prefix-bank smoke (2,4 prompts) |
