"""Perplexity on the staff windows: batch 1, no state across windows, FP32 log-softmax of FP16 logits."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def window_nll(model, token_ids: list[int], lo: int, hi: int, device) -> float:
    """Sum NLL of targets token_ids[lo:hi], each predicted from token_ids[:t] within this window only."""
    ids = torch.tensor(token_ids[:hi], dtype=torch.long, device=device)[None]
    logits = model(ids, use_cache=False).logits[0]  # [hi, V] fp16
    logp = torch.log_softmax(logits[lo - 1:hi - 1].float(), dim=-1)  # predictions for targets lo..hi-1
    tgt = ids[0, lo:hi]
    return float(-logp.gather(-1, tgt[:, None]).double().sum())


def evaluate_windows(model, windows: list[dict], device, progress_every: int = 0) -> tuple[list[dict], dict]:
    rows = []
    for i, w in enumerate(windows):
        lo, hi = w["score_target_offsets"]
        nll = window_nll(model, w["token_ids"], lo, hi, device)
        rows.append({"window_id": w["window_id"], "source": w.get("source"), "source_row": w["source_row"],
                     "token_offset": w.get("token_offset"), "scored_tokens": hi - lo, "sum_nll": nll})
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ppl {i + 1}/{len(windows)}", flush=True)
    return rows, summarize(rows)


def summarize(rows: list[dict]) -> dict:
    n = sum(r["scored_tokens"] for r in rows)
    s = sum(r["sum_nll"] for r in rows)
    finite = all(math.isfinite(r["sum_nll"]) for r in rows)
    return {"N": n, "windows": len(rows), "sum_nll": s, "mean_nll": s / n, "ppl": math.exp(s / n),
            "all_finite": finite}
