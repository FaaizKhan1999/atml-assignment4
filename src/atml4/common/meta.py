"""Run metadata collected into every result file's `meta` block."""
from __future__ import annotations

import datetime as _dt
import os
import platform
import shlex
import socket
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

SUBMODULE = Path("external/atml-pa4-datasets")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    env = os.environ.get("ATML4_ROOT")
    if env:
        return Path(env).resolve()
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "configs" / "models.yaml").exists():
            return parent
    return Path.cwd().resolve()


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> dict:
    root = repo_root()
    commit = _git(["rev-parse", "HEAD"], root) or os.environ.get("ATML4_GIT_COMMIT")
    status = _git(["status", "--porcelain", "--untracked-files=no"], root)
    sub = _git(["rev-parse", "HEAD"], root / SUBMODULE)
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "submodule_path": str(SUBMODULE),
        "submodule_commit": sub,
    }


def load_models_config() -> dict:
    import yaml

    with open(repo_root() / "configs" / "models.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["models"]


def model_revision(key_or_id: str) -> tuple[str, str]:
    """Return (hf_id, revision) for a key ('base'/'target'/'draft') or HF id from configs/models.yaml."""
    for key, spec in load_models_config().items():
        if key_or_id in (key, spec["id"]):
            return spec["id"], spec["revision"]
    raise KeyError(f"{key_or_id} not in configs/models.yaml; refusing to load an unpinned model")


def _versions() -> dict:
    out: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "huggingface_hub", "datasets", "numpy", "scipy", "peft", "accelerate"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def sdpa_info() -> dict:
    """SDPA backend enable-flags and build/runtime availability (not which kernel a call dispatches to)."""
    try:
        import torch
    except ImportError:
        return {}
    b = torch.backends.cuda
    info: dict[str, Any] = {}
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        fn = getattr(b, name, None)
        info[name] = bool(fn()) if fn else None
    for name in ("is_flash_attention_available",):
        fn = getattr(b, name, None)
        try:
            info[name] = bool(fn()) if fn else None
        except Exception as e:  # pragma: no cover
            info[name] = f"error: {e}"
    return info


def cuda_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False, "torch_cuda_build": getattr(torch.version, "cuda", None)}
    devs = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devs.append({"index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
                     "total_memory_bytes": p.total_memory})
    driver = None
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:
        pass
    return {"available": True, "devices": devs, "torch_cuda_build": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "driver": driver}


def collect_meta(*, seed: int | None = None, dtype: Any = None, models: Iterable[str] = (),
                 command: list[str] | str | None = None, extra: dict | None = None) -> dict:
    """Collect reproducibility metadata.

    models: keys or HF ids from configs/models.yaml actually used by this run.
    command: exact command line; defaults to sys.argv of this process.
    """
    if command is None:
        command = [Path(sys.executable).name, *sys.argv]
    cmd = command if isinstance(command, str) else shlex.join(command)
    used = []
    for m in models:
        mid, rev = model_revision(m)
        used.append({"id": mid, "revision": rev})
    meta = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": cmd,
        "cwd": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "kaggle": {"kernel_slug": os.environ.get("ATML4_KERNEL_SLUG"),
                   "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")},
        "git": git_info(),
        "versions": _versions(),
        "cuda": cuda_info(),
        "sdpa": sdpa_info(),
        "dtype": str(dtype) if dtype is not None else None,
        "seed": seed,
        "models": used,
    }
    if extra:
        meta["extra"] = extra
    return meta
