"""Question-aware importance scores (C2(ii) and the hybrid): a SIMPLIFIED, fixed SnapKV-inspired policy
(Li et al., 2024), not a reproduction of the published head-specific algorithm (no pooling, no per-head selection,
one global token set).

The question suffix is run through the WORKING SDPA model on top of the full FP16 context cache (no eager path).
Forward hooks capture every layer's q_proj output; RoPE is applied at the suffix's absolute positions L..L+S-1
with the model's own rotary embedding; scores are masked FP32 softmax attention against the rotary-positioned
cached keys (context keys 0..L-1 plus the causal suffix keys, so the softmax is exactly the model's attention
pattern up to FP16 vs FP32 arithmetic). GQA: query head h reads KV head h // (H / H_kv) (transformers repeat_kv).
Aggregation (fixed a priori): mean over suffix query positions and query heads, then mean over layers; only the
L context columns are used for ranking. No parameter is tuned, so no development prompts are needed.
"""
from __future__ import annotations

import math

import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from .kvcache import forward_step, make_cache


@torch.no_grad()
def question_scores(model, Ks: list, Vs: list, suffix_ids: list[int], device) -> dict:
    cfg = model.config
    L = Ks[0].shape[2]
    S = len(suffix_ids)
    H, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
    d = Ks[0].shape[-1]
    caps = {}
    hooks = [layer.self_attn.q_proj.register_forward_hook(
        lambda m, a, o, i=i: caps.__setitem__(i, o.detach())) for i, layer in enumerate(model.model.layers)]
    cache = make_cache(Ks, Vs)
    try:
        logits = forward_step(model, cache, suffix_ids, L, device)
    finally:
        for h in hooks:
            h.remove()
    pos = torch.arange(L, L + S, device=device)[None]
    dummy = torch.zeros(1, S, d, device=device, dtype=Ks[0].dtype)
    cos, sin = model.model.rotary_emb(dummy, pos)
    rep = H // Hkv
    causal = torch.ones(S, L + S, dtype=torch.bool, device=device)
    causal[:, L:] = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
    per_layer = []
    for i, layer in enumerate(cache.layers):
        q = caps[i].view(1, S, H, d).transpose(1, 2)  # [1, H, S, d]
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        k = layer.keys[0]  # [Hkv, L + S, d] post-RoPE (context keys unchanged, suffix keys appended)
        kq = k.repeat_interleave(rep, dim=0).float()  # head h -> kv head h // rep
        att = torch.einsum("hsd,htd->hst", q[0].float(), kq) / math.sqrt(d)
        att = att.masked_fill(~causal, float("-inf")).softmax(-1)
        per_layer.append(att[:, :, :L].mean(dim=(0, 1)))
    scores = torch.stack(per_layer).mean(0)
    return {"scores": scores, "per_layer": torch.stack(per_layer), "suffix_logits": logits}
