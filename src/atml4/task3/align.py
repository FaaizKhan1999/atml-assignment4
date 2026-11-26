"""Step 5: distribution alignment of drafts vs the target on common prefix banks.

Per saved prefix (bank, prompt, position p = the p-th response token is predicted from the p-1 saved tokens):
forward KL(p||q), reverse KL(q||p), TV = 0.5 sum |p - q|, expected one-token acceptance 1 - TV, and greedy agreement
argmax p == argmax q; FP32 softmax over the full 151,936-logit vocabulary, T = 1. All positions of one source
sequence are prefixes of the same sequence, so ONE forward over prompt + longest saved prefix gives every position
(asserted: every saved prefix is a prefix of the longest one). Aggregates are token-weighted (each saved prefix
counts once in the pooled denominator).
"""
from __future__ import annotations

from collections import defaultdict

import torch

POS_BINS = {"[1,16]": (1, 16), "[17,32]": (17, 32), "[33,64]": (33, 64), "[65,192]": (65, 192)}


def pos_bin(p: int) -> str:
    return next(k for k, (a, b) in POS_BINS.items() if a <= p <= b)


@torch.no_grad()
def _logits_at(model, prompt_ids: list[int], prefix: list[int], positions: list[int], device) -> torch.Tensor:
    x = torch.tensor([prompt_ids + prefix], device=device)
    lg = model(input_ids=x, use_cache=False).logits[0]
    idx = torch.tensor([len(prompt_ids) - 1 + (p - 1) for p in positions], device=device)
    return lg[idx].float()


def metrics(p_logits: torch.Tensor, q_logits: torch.Tensor) -> dict:
    lp, lq = torch.log_softmax(p_logits, -1), torch.log_softmax(q_logits, -1)
    p, q = lp.exp(), lq.exp()
    tv = 0.5 * (p - q).abs().sum(-1)
    return {"fkl": (p * (lp - lq)).sum(-1), "rkl": (q * (lq - lp)).sum(-1), "tv": tv, "acc1": 1 - tv,
            "greedy_agree": (p.argmax(-1) == q.argmax(-1)).float()}


def group_bank(records: list[dict]) -> dict:
    g = defaultdict(list)
    for r in records:
        g[(r["bank"], r["prompt_id"])].append(r)
    out = {}
    for key, rs in g.items():
        rs = sorted(rs, key=lambda r: r["position"])
        longest = max((r["response_prefix_token_ids"] for r in rs), key=len)
        for r in rs:
            pre = r["response_prefix_token_ids"]
            assert pre == longest[:len(pre)] and len(pre) == r["position"] - 1, key
        out[key] = {"positions": [r["position"] for r in rs], "prefix": longest, "domain": rs[0]["domain"],
                    "prompt": rs[0]["prompt"], "records": rs}
    return out


@torch.no_grad()
def evaluate(target, drafts: dict, groups: dict, prompt_ids_fn, workload_of: dict, device) -> list[dict]:
    """Returns one row per (bank, prompt, position, draft)."""
    rows = []
    for (bank, pid), g in groups.items():
        pids = prompt_ids_fn(g["prompt"])
        need = max(g["positions"]) - 1
        prefix = g["prefix"][:need]
        pl = _logits_at(target, pids, prefix, g["positions"], device)
        for name, dm in drafts.items():
            ql = _logits_at(dm, pids, prefix, g["positions"], device)
            m = metrics(pl, ql)
            for i, pos in enumerate(g["positions"]):
                rows.append({"bank": bank, "prompt_id": pid, "workload": workload_of[pid], "position": pos,
                             "bin": pos_bin(pos), "draft": name, **{k: float(v[i]) for k, v in m.items()}})
    return rows


def aggregate(rows: list[dict], keys=("bank", "draft")) -> dict:
    acc = defaultdict(lambda: defaultdict(float))
    n = defaultdict(int)
    for r in rows:
        k = tuple(r[x] for x in keys)
        n[k] += 1
        for m in ("fkl", "rkl", "tv", "acc1", "greedy_agree"):
            acc[k][m] += r[m]
    return {"|".join(map(str, k)): {"n_prefixes": n[k], **{m: v / n[k] for m, v in acc[k].items()}} for k in acc}
