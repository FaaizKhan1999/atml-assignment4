"""Seeding. Default reference seed is 6304 (assignment-wide)."""
from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 6304


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> int:
    """Seed python, numpy and torch (CPU + all CUDA devices).

    deterministic=True additionally sets:
      * torch.backends.cudnn.deterministic=True, benchmark=False
      * torch.use_deterministic_algorithms(True, warn_only=True)  -- warn_only so that ops without a
        deterministic implementation (e.g. some scatter/index_add backward kernels) warn instead of crash
      * CUBLAS_WORKSPACE_CONFIG=:4096:8 (required for deterministic cuBLAS; must precede the first cuBLAS call)
    Flash/mem-efficient SDPA backward is not bitwise deterministic on all GPUs; timing benchmarks may pass
    deterministic=False to avoid slowing kernels.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
