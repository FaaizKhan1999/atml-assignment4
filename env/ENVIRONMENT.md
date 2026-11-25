# Environment

- Direct pins: `env/requirements.txt` (transformers==4.57.1, huggingface_hub==0.36.2 as required by the
  assignment; `datasets>=3,<5` from the course data repo's requirements-public.txt). torch is NOT pinned: the
  Kaggle image's torch is used.
- Lock: `env/requirements-lock.txt` was frozen with `pip freeze` inside the Kaggle GPU smoke kernel
  (`scripts/gpu_smoke.py`) because **no course environment lock file was provided**.
- Model revisions: `configs/models.yaml`. Dataset source revisions: `configs/local_source_revisions.json`
  (copied from the data-preparation kernel's `data/local_source_revisions.json`).

## Kaggle GPU runtime (from the smoke kernel)

Source: `results/env/gpu_smoke.json` (kernel focusedkognition/atml4-gpu-smoke, commit 40daf27, 2026-11-25).

| item | value |
|---|---|
| Python | 3.13.16 |
| torch | 2.11.0+cu130 (Kaggle image, not pinned) |
| CUDA (torch build) | 13.0 |
| NVIDIA driver / CUDA (nvidia-smi) | 580.178.04 / 13.0 |
| cuDNN (`torch.backends.cudnn.version()`) | 92700; lock contains nvidia-cudnn-cu12==9.27.0.42 and nvidia-cudnn-cu13==9.19.0.56 |
| transformers / huggingface_hub / datasets / numpy | 4.57.1 / 0.36.2 / 4.8.5 / 2.1.3 |
| GPU | 2x Tesla T4 (requested `--accelerator NvidiaTeslaT4`), compute capability 7.5, 15360 MiB each |

SDPA, causal FP16, B=2 H=4 L=1024 d=64 (forced via `torch.nn.attention.sdpa_kernel`):

| backend | result |
|---|---|
| FLASH_ATTENTION | RuntimeError: No available kernel (T4 is sm75) |
| EFFICIENT_ATTENTION | OK, median 0.282 ms (10 reps) |
| CUDNN_ATTENTION | RuntimeError: No available kernel |
| MATH | OK, median 2.726 ms (10 reps) |

Default dispatch: EFFICIENT_ATTENTION (`torch._fused_sdp_choice`), confirmed by profiler kernel
`fmha_cutlassF_f16_aligned_64x64_rf_sm75` (PyTorchMemEffAttention).
