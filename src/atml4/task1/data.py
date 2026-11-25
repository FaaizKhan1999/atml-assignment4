"""Loaders for the staff-prepared Task 1 data (external/atml-pa4-datasets/data/public_locked)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from atml4.common.io import iter_jsonl
from atml4.common.meta import repo_root

DATA_DIR = Path("external/atml-pa4-datasets/data/public_locked")
FILES = {
    "calibration": "c4_calibration_locked.jsonl.gz",
    "development": "c4_development_locked.jsonl.gz",
    "wikitext2": "wikitext2_test_windows_locked.jsonl.gz",
    "c4": "c4_validation_windows_locked.jsonl.gz",
    "hellaswag": "hellaswag_val_locked.jsonl.gz",
}
HELLASWAG_DIAGNOSTIC_N = 512  # first 512 of the 2,000 supplied IDs (course README / PDF)


def data_path(key: str) -> Path:
    return repo_root() / DATA_DIR / FILES[key]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(key: str) -> list[dict]:
    return list(iter_jsonl(data_path(key)))


def load_windows(suite: str) -> list[dict]:
    """Perplexity windows; adds `window_id` = file order index. Validates the staff fields."""
    rows = load(suite)
    for i, r in enumerate(rows):
        lo, hi = r["score_target_offsets"]
        if hi - lo != r["scored_tokens"] or not (1 <= lo < hi <= len(r["token_ids"])):
            raise ValueError(f"{suite} window {i}: inconsistent score_target_offsets")
        r["window_id"] = i
    return rows


def selection_record(keys) -> dict:
    """Dataset-selection identifiers for result meta: file sha256, count, data lock."""
    out = {}
    for k in keys:
        p = data_path(k)
        out[k] = {"file": str(DATA_DIR / FILES[k]), "sha256": file_sha256(p)}
    lock = repo_root() / "configs/local_source_revisions.json"
    out["source_revisions"] = json.loads(lock.read_text())["revisions"] if lock.exists() else None
    return out
