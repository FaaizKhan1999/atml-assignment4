"""Atomic JSON/JSONL result writing. Every result carries a `meta` block."""
from __future__ import annotations

import gzip
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def _default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, torch.dtype):
            return str(o)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def _atomic_write(path: Path, write_fn) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as raw:
            fh = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) if path.suffix == ".gz" else raw
            write_fn(fh)
            if fh is not raw:
                fh.close()
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def _dumps(obj: Any, **kw) -> str:
    return json.dumps(obj, default=_default, allow_nan=True, ensure_ascii=False, **kw)


def save_result(path, payload: Any, meta: dict) -> Path:
    """Write a result file atomically.

    *.json / *.json.gz  -> {"meta": meta, "payload": payload}
    *.jsonl / *.jsonl.gz -> first line {"meta": meta}, then one JSON object per item of `payload` (iterable).
    """
    if meta is None or "command" not in meta:
        raise ValueError("meta must be a collect_meta() dict including 'command'")
    path = Path(path)
    name = path.name[:-3] if path.name.endswith(".gz") else path.name
    if name.endswith(".jsonl"):
        def w(fh):
            fh.write((_dumps({"meta": meta}) + "\n").encode())
            for row in payload:
                fh.write((_dumps(row) + "\n").encode())
    elif name.endswith(".json"):
        def w(fh):
            fh.write(_dumps({"meta": meta, "payload": payload}, indent=1).encode())
    else:
        raise ValueError(f"unsupported extension: {path}")
    return _atomic_write(path, w)


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def load_result(path) -> tuple[dict, Any]:
    """Return (meta, payload) for a .json result written by save_result."""
    with _open(Path(path)) as f:
        obj = json.load(f)
    return obj["meta"], obj["payload"]


def load_jsonl_result(path) -> tuple[dict, list]:
    """Return (meta, rows) for a .jsonl result written by save_result."""
    rows = list(iter_jsonl(path))
    if not rows or set(rows[0].keys()) != {"meta"}:
        raise ValueError(f"{path}: first line is not a meta block")
    return rows[0]["meta"], rows[1:]


def iter_jsonl(path) -> Iterable[dict]:
    """Iterate any JSONL / JSONL.gz file (also the course data files)."""
    with _open(Path(path)) as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{i}: {e}") from e


def is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
