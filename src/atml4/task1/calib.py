"""Calibration statistics on eligible-projection INPUTS and a streaming relative output-error helper."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .model import eligible_projections


def calibration_batches(records: list[dict], batch_size: int, device):
    """Yield [B, 512] LongTensors of the staff calibration token windows in file order."""
    for i in range(0, len(records), batch_size):
        ids = [r["token_ids"] for r in records[i:i + batch_size]]
        if len({len(x) for x in ids}) != 1:
            raise ValueError("calibration windows must have equal length")
        yield torch.tensor(ids, dtype=torch.long, device=device)


@torch.no_grad()
def collect_input_stats(model, batches) -> dict[str, dict[str, np.ndarray]]:
    """Per eligible projection and input channel j (FP32 accumulation):
    sumsq_j = sum_t x_tj^2 (so ||X_j||_2 = sqrt(sumsq_j)), sumabs_j = sum_t |x_tj|, count = #tokens."""
    projs = eligible_projections(model)
    acc = {p.qualname: {"sumsq": torch.zeros(p.shape[1], dtype=torch.float32, device=p.module.weight.device),
                        "sumabs": torch.zeros(p.shape[1], dtype=torch.float32, device=p.module.weight.device),
                        "count": 0} for p in projs}
    hooks = []

    def make(q):
        def hook(mod, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            acc[q]["sumsq"] += (x * x).sum(0)
            acc[q]["sumabs"] += x.abs().sum(0)
            acc[q]["count"] += x.shape[0]
        return hook

    for p in projs:
        hooks.append(p.module.register_forward_pre_hook(make(p.qualname)))
    try:
        for b in batches:
            model(b, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return {q: {"sumsq": v["sumsq"].cpu().numpy(), "sumabs": v["sumabs"].cpu().numpy(),
                "count": np.int64(v["count"])} for q, v in acc.items()}


def save_stats(path, stats: dict, source_rows: list[int], seed: int) -> None:
    arrays = {}
    for q, v in stats.items():
        for k, a in v.items():
            arrays[f"{q}.{k}"] = np.asarray(a)
    arrays["_source_rows"] = np.asarray(source_rows, dtype=np.int64)
    arrays["_seed"] = np.int64(seed)
    np.savez_compressed(path, **arrays)


def load_stats(path) -> tuple[dict, np.ndarray]:
    z = np.load(path)
    out: dict[str, dict] = {}
    for key in z.files:
        if key.startswith("_"):
            continue
        q, k = key.rsplit(".", 1)
        out.setdefault(q, {})[k] = z[key]
    return out, z["_source_rows"]


@torch.no_grad()
def relative_output_error(model, batches, new_weight: Callable[[str], torch.Tensor | None]) -> dict[str, dict]:
    """Isolated per-projection relative output error on calibration inputs (NOT yet used by any condition).

    The model runs UNCHANGED; for each eligible projection with replacement W' = new_weight(qualname) (None =
    skip), its actual input X (from the unmodified upstream network) is streamed through a hook and
      num += ||(W - W') X||_F^2,  den += ||W X||_F^2   (no bias; FP32 matmuls; float64 accumulation)
    rel = num / den. Inputs are never stored. Interactions between modified layers are NOT captured.
    """
    projs = eligible_projections(model)
    res, hooks = {}, []

    def make(q, w2):
        def hook(mod, args):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            w = mod.weight.float()
            res[q]["num"] += float((F.linear(x, w - w2.to(w.device).float()) ** 2).sum().double())
            res[q]["den"] += float((F.linear(x, w) ** 2).sum().double())
        return hook

    for p in projs:
        w2 = new_weight(p.qualname)
        if w2 is None:
            continue
        if tuple(w2.shape) != p.shape:
            raise ValueError(f"shape mismatch for {p.qualname}")
        res[p.qualname] = {"num": 0.0, "den": 0.0}
        hooks.append(p.module.register_forward_pre_hook(make(p.qualname, w2)))
    try:
        for b in batches:
            model(b, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    for v in res.values():
        v["rel"] = v["num"] / v["den"] if v["den"] > 0 else float("nan")
    return res
