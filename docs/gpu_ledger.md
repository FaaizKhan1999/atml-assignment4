# GPU ledger

One row per Kaggle kernel run (appended by `scripts/kaggle_run.py fetch`). Wall time is the bootstrap's own wall clock (excludes Kaggle queue/startup). GPU-hours = wall hours for GPU kernels, 0 for CPU kernels.

**Running total GPU-hours: 0.000**

| date (UTC) | kernel | accelerator (nvidia-smi) | wall time (h) | GPU-hours | status | commit | purpose |
|---|---|---|---|---|---|---|---|
| 2026-11-25T16:15:27+00:00 | focusedkognition/atml4-data-prep | none (CPU) | 0.020 | 0.000 | ok | b3c8016772 | Task data prep: setup_data.py --public + check_setup.py |
