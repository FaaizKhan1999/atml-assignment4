"""Batched temperature-1 sampling for NON-TIMING generation (teacher-response collection, on-policy rollouts,
own-student prefix banks, standalone accuracy).

Left padding with an attention mask and explicit position_ids (cumsum of the mask), FP32 softmax over the full
vocabulary (T = 1, no top-k / top-p / repetition penalty), one torch.Generator PER ROW seeded with that row's
recorded seed, so a row's draws do not depend on the other rows' random numbers. A row stops at its first EOS
(EOS kept) or at its cap. Batch composition can change FP16 numerics slightly (padding); timing runs never use this.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache


@torch.no_grad()
def batched_sample(model, prompts: list[list[int]], caps: list[int], seeds: list[int], eos_ids, device,
                   pad_id: int, temperature: float = 1.0) -> list[dict]:
    B = len(prompts)
    eos = torch.tensor(sorted(set(eos_ids)), device=device)
    Lp = max(len(p) for p in prompts)
    ids = torch.full((B, Lp), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros(B, Lp, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        ids[i, Lp - len(p):] = torch.tensor(p, device=device)
        mask[i, Lp - len(p):] = 1
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    gens = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]
    cache = DynamicCache()
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True,
                logits_to_keep=1)
    logits = out.logits[:, -1].float()
    nxt_pos = pos[:, -1] + 1
    res = [[] for _ in range(B)]
    done = [False] * B
    stop = ["cap"] * B
    capt = torch.tensor(caps, device=device)
    step = 0
    while not all(done):
        p = torch.softmax(logits / temperature, -1)
        toks = torch.full((B,), pad_id, dtype=torch.long, device=device)
        for i in range(B):
            if done[i]:
                continue
            t = int(torch.multinomial(p[i], 1, generator=gens[i]))
            toks[i] = t
            res[i].append(t)
            if bool((eos == t).any()):
                done[i], stop[i] = True, "eos"
            elif len(res[i]) >= caps[i]:
                done[i] = True
        step += 1
        if all(done):
            break
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=device)], 1)
        out = model(input_ids=toks[:, None], attention_mask=mask, position_ids=nxt_pos[:, None],
                    past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1].float()
        nxt_pos = nxt_pos + 1
    del capt
    return [{"tokens": r, "stop": s} for r, s in zip(res, stop)]
