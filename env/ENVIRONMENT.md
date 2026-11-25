# Environment

- Direct pins: `env/requirements.txt` (transformers==4.57.1, huggingface_hub==0.36.2 as required by the
  assignment; `datasets>=3,<5` from the course data repo's requirements-public.txt). torch is NOT pinned: the
  Kaggle image's torch is used.
- Lock: `env/requirements-lock.txt` was frozen with `pip freeze` inside the Kaggle GPU smoke kernel
  (`scripts/gpu_smoke.py`) because **no course environment lock file was provided**.
- Model revisions: `configs/models.yaml`. Dataset source revisions: `configs/local_source_revisions.json`
  (copied from the data-preparation kernel's `data/local_source_revisions.json`).

## Kaggle GPU runtime (from the smoke kernel)

PENDING: filled from results/env/gpu_smoke.json after the first GPU smoke run.
