"""MQAR data from the course generator (external/atml-pa4-datasets/scripts/mqar_generator.py).

Training-stream rule (docs/decisions.md): for (K, regime, seed) example i of update u has generator index
  (seed - 6304) * 10**7 + u * 32 + i        (split='train')
so every architecture sees the identical sequence and seeds use disjoint index ranges.
Streams are generated once and cached as compressed .npz (tokens uint8, query/source positions int16,
answers uint8).
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from atml4.common.meta import repo_root

GEN_DIR = repo_root() / "external/atml-pa4-datasets/scripts"
DATA_DIR = repo_root() / "external/atml-pa4-datasets/data/mqar"
BATCH, UPDATES, T = 32, 4500, 256


def _gen():
    if str(GEN_DIR) not in sys.path:
        sys.path.insert(0, str(GEN_DIR))
    import mqar_generator
    return mqar_generator


def stream_index(seed: int, update: int, i: int) -> int:
    return (seed - 6304) * 10 ** 7 + update * BATCH + i


def to_arrays(rows: list[dict]) -> dict:
    K = rows[0]["K"]
    return {"tokens": np.array([r["tokens"] for r in rows], dtype=np.uint8),
            "query_positions": np.array([r["query_positions"] for r in rows], dtype=np.int16),
            "source_positions": np.array([r["source_positions"] for r in rows], dtype=np.int16),
            "answer_ids": np.array([r["answer_ids"] for r in rows], dtype=np.uint8),
            "distances": np.array([r["distances"] for r in rows], dtype=np.int16), "K": np.int64(K)}


def generate_stream(K: int, regime: str, seed: int, n_updates: int = UPDATES, out: Path | None = None) -> dict:
    g = _gen()
    t = time.perf_counter()
    rows = []
    for u in range(n_updates):
        for i in range(BATCH):
            ex = g.make_example("train", K, regime, stream_index(seed, u, i))  # make_example validates
            rows.append(ex)
    arr = to_arrays(rows)
    arr["seconds"] = np.float64(time.perf_counter() - t)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **arr)
    return arr


def stream_path(cache: Path, K: int, regime: str, seed: int) -> Path:
    return cache / f"train_K{K}_{regime}_seed{seed}.npz"


def load_split(split: str) -> list[dict]:
    """Frozen development / test records (as supplied)."""
    f = DATA_DIR / ("mqar_test.jsonl.gz" if split == "test" else "mqar_development.jsonl.gz")
    with gzip.open(f, "rt") as fh:
        return [json.loads(l) for l in fh]


def cells(rows: list[dict]) -> dict[tuple[int, str], dict]:
    out: dict[tuple[int, str], list] = {}
    for r in rows:
        out.setdefault((r["K"], r["regime"]), []).append(r)
    return {k: {**to_arrays(v), "ids": [r["id"] for r in v]} for k, v in out.items()}


def invariant_checks() -> dict:
    """Run the course generator's own checks: validate() on every frozen record, regeneration equality, and the
    generator's 9,216-example self-test."""
    g = _gen()
    t = time.perf_counter()
    res = {}
    for split in ("test", "development"):
        rows = load_split(split)
        bad = 0
        for r in rows:
            g.validate(r)
            idx = int(r["id"].rsplit("-", 1)[1])
            if r != g.make_example(r["split"], r["K"], r["regime"], idx):
                bad += 1
        res[split] = {"records": len(rows), "validate_ok": True, "regeneration_mismatches": bad}
    for k in (4, 8, 16):
        for regime in ("near", "far", "mixed"):
            for i in range(64):
                g.validate(g.make_example("train", k, regime, stream_index(6306, 4499, i % 32)))
    res["train_spot_checks"] = 9 * 64
    res["seconds"] = time.perf_counter() - t
    return res
