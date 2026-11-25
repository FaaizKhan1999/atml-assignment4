"""Training / evaluation loop for Task 2 Part B (fixed PDF protocol)."""
from __future__ import annotations

import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .mqar_data import BATCH, UPDATES
from .mqar_model import MQARModel, lr_at, n_params, query_loss

EVAL_AT = (500, 1000, 1500, 2000, 3000, 4500)
LOG_EVERY = 25
DIST_BINS = [(0, 32), (32, 64), (64, 128), (128, 256)]


def batch_tensors(arr: dict, sl, device):
    return (torch.from_numpy(arr["tokens"][sl].astype(np.int64)).to(device),
            torch.from_numpy(arr["query_positions"][sl].astype(np.int64)).to(device),
            torch.from_numpy(arr["answer_ids"][sl].astype(np.int64)).to(device))


@torch.no_grad()
def evaluate(model, cell_arrays: dict, device, batch: int = 256, return_queries: bool = False) -> dict:
    model.eval()
    out = {}
    for (K, regime), arr in sorted(cell_arrays.items()):
        n = arr["tokens"].shape[0]
        preds = []
        for i in range(0, n, batch):
            tok, qp, ans = batch_tensors(arr, slice(i, i + batch), device)
            preds.append((model(tok, qp).argmax(-1) + 128).cpu().numpy())
        pred = np.concatenate(preds)
        gold = arr["answer_ids"].astype(np.int64)
        # distance from the generator's saved source position (never inferred from the query location)
        dist = arr["query_positions"].astype(np.int64) - arr["source_positions"].astype(np.int64)
        assert (dist == arr["distances"]).all()
        corr = pred == gold
        rec = {"K": K, "regime": regime, "n_seq": n, "n_queries": int(corr.size), "acc": float(corr.mean()),
               "chance_label_space": 1 / 64, "chance_1_over_K": 1 / K,
               "dist_bins": {f"[{a},{b})": {"n": int(((dist >= a) & (dist < b)).sum()),
                                           "acc": float(corr[(dist >= a) & (dist < b)].mean())
                                           if ((dist >= a) & (dist < b)).any() else None} for a, b in DIST_BINS}}
        if return_queries:
            rec["_queries"] = {"pred": pred, "gold": gold, "dist": dist, "ids": arr["ids"]}
        out[f"K{K}_{regime}"] = rec
    model.train()
    return out


def overall(evals: dict) -> float:
    n = sum(v["n_queries"] for v in evals.values())
    return sum(v["acc"] * v["n_queries"] for v in evals.values()) / n


def train_run(arch: str, K: int, regime: str, seed: int, stream: dict, dev_cells: dict, device,
              log=print) -> tuple[MQARModel, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MQARModel(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_at(0), weight_decay=0.01)
    curve, dev_curve = [], []
    run_loss, run_n = 0.0, 0
    t_train = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    nonfinite = 0
    for u in range(UPDATES):
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(u)
        tok, qp, ans = batch_tensors(stream, slice(u * BATCH, (u + 1) * BATCH), device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        loss = query_loss(model(tok, qp), ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_train += time.perf_counter() - t0
        lv = float(loss)
        if not math.isfinite(lv):
            nonfinite += 1
        run_loss += lv
        run_n += 1
        if (u + 1) % LOG_EVERY == 0:
            curve.append({"update": u + 1, "train_loss": run_loss / run_n, "lr": lr_at(u), "grad_norm": float(gnorm)})
            run_loss, run_n = 0.0, 0
        if (u + 1) in EVAL_AT:
            ev = evaluate(model, dev_cells, device)
            dev_curve.append({"update": u + 1, "dev": {k: {"acc": v["acc"]} for k, v in ev.items()},
                              "dev_trained_cell_acc": ev[f"K{K}_{regime}"]["acc"]})
            log(f"  [{arch} K{K} {regime} s{seed}] u={u + 1} loss={curve[-1]['train_loss']:.4f} "
                f"dev({regime},K{K})={ev[f'K{K}_{regime}']['acc']:.4f}")
    info = {"arch": arch, "K": K, "regime": regime, "seed": seed, "params": n_params(model),
            "train_curve": curve, "dev_curve": dev_curve, "train_seconds_steps_only": t_train,
            "mean_step_ms": 1000 * t_train / UPDATES, "nonfinite_losses": nonfinite,
            "peak_alloc_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None}
    return model, info


def write_queries(path: Path, evals: dict) -> None:
    with gzip.open(path, "wt") as f:
        for cell, v in evals.items():
            q = v.pop("_queries")
            for si, sid in enumerate(q["ids"]):
                for j in range(q["pred"].shape[1]):
                    f.write(json.dumps({"cell": cell, "id": sid, "q": j, "K": v["K"], "regime": v["regime"],
                                        "distance": int(q["dist"][si, j]), "gold": int(q["gold"][si, j]),
                                        "pred": int(q["pred"][si, j])}) + "\n")
