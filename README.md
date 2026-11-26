# ATML PA4 (EE-5102/CS-6304, Fall 2026) — code

Code for Programming Assignment 4 (model compression, efficient attention / KV memory, draft distillation for
speculative decoding). Status: infrastructure only.

## Layout
- `src/atml4/common/` seed, timing, meta, io, stats (bootstrap, chi-square), plotting (NeurIPS style)
- `src/atml4/task{1,2,3}/` task code; `scripts/` entry points; `configs/` pinned model and data revisions
- `results/` small machine-readable outputs (each with a `meta` block holding the exact command)
- `external/atml-pa4-datasets/` course data repository (git submodule, pinned commit)
- `kaggle/` + `scripts/kaggle_run.py` Kaggle kernel runner; `docs/` GPU ledger, decisions, deviations

## Setup
```bash
git clone --recurse-submodules https://github.com/FaaizKhan1999/atml-assignment4.git
cd atml-assignment4
pip install -r env/requirements.txt   # plus the torch of your CUDA runtime
pytest -q
```

## Data preparation (Kaggle CPU kernel)
```bash
python scripts/kaggle_run.py launch data-prep --cpu --wait --output external/atml-pa4-datasets/data \
  --cmd "cd external/atml-pa4-datasets && python scripts/setup_data.py --public && python scripts/check_setup.py"
```

## GPU smoke test (Kaggle)
```bash
python scripts/kaggle_run.py launch gpu-smoke --gpu --wait --source <user>/atml4-data-prep \
  --output env/requirements-lock.txt --cmd "python scripts/gpu_smoke.py"
```

## Artifacts kept outside git
- Task 2 Part B MQAR checkpoints (39 runs, `results/task2/partB/ckpt/<run>.pt`, ~1.7 MB each) are NOT committed.
  They are in the output of Kaggle kernel `focusedkognition/atml4-t2-ab` (run at commit d651fac), and the B3
  curriculum checkpoints (`window_K16_curriculum_s*.pt`) in the output of `focusedkognition/atml4-q3proj-b3`.
  Per-query test predictions for every run are committed (`results/task2/partB/queries/*.jsonl.gz`).

## Attribution of reused code
None so far.
