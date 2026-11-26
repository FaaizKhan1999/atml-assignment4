"""Cache mechanics for edited KV caches (transformers 4.57.1, Qwen2, SDPA).

How Qwen2 consumes positions (read from transformers 4.57.1 models/qwen2/modeling_qwen2.py + cache_utils.py +
masking_utils.py):
  * Qwen2Model.forward: if cache_position is None it is arange(past_seen, past_seen + n) with past_seen =
    cache.get_seq_length() (= PHYSICAL number of stored entries of a DynamicLayer, keys.shape[-2]); if position_ids
    is None it is cache_position[None].
  * RoPE uses ONLY position_ids: rotary_emb(hidden, position_ids) -> cos/sin applied to the new q and k before
    DynamicLayer.update, which concatenates them after the stored (already rotated) keys; stored keys are never
    re-rotated.
  * The causal mask uses cache_position and the cache's mask sizes: DynamicLayer.get_mask_sizes(cache_position)
    -> kv_length = physical_len + n, kv_offset = 0; query i may see physical key j iff j <= cache_position[i].
  * A 4D attention mask is passed through unchanged (masking_utils._preprocess_mask_arguments early exit).
Therefore an edited cache with P physical entries must be extended with cache_position = arange(P, P + n)
(physical, consistent with the mask/cache length) and position_ids = the ORIGINAL absolute positions (continuing
from the full context length L), and a 2D attention_mask of ones with length P + n.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache

from . import kvquant as KQ


def make_cache(keys: list, values: list) -> DynamicCache:
    """Plain DynamicLayers (no config -> no sliding layers) holding FP16 tensors [1, H, P, d]."""
    return DynamicCache(ddp_cache_data=[(k.clone(), v.clone()) for k, v in zip(keys, values)])


@torch.no_grad()
def prefill(model, ids: list[int], device) -> tuple[list, list, torch.Tensor]:
    x = torch.tensor([ids], device=device)
    cache = DynamicCache()
    out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
    ks = [l.keys for l in cache.layers]
    vs = [l.values for l in cache.layers]
    return ks, vs, out.logits[0, -1].float()


@torch.no_grad()
def forward_step(model, cache: DynamicCache, ids: list[int], abs_start: int, device) -> torch.Tensor:
    """Append ids at absolute positions abs_start.. with physical cache_position; returns logits [n, V] (fp32)."""
    n = len(ids)
    P = cache.get_seq_length()
    x = torch.tensor([ids], device=device)
    cache_position = torch.arange(P, P + n, device=device)
    position_ids = torch.arange(abs_start, abs_start + n, device=device)[None]
    attn = torch.ones(1, P + n, dtype=torch.long, device=device)
    out = model(input_ids=x, past_key_values=cache, position_ids=position_ids, cache_position=cache_position,
                attention_mask=attn, use_cache=True)
    return out.logits[0].float()


class RunningQuant:
    """New tokens stay FP16 until a complete 64-token group forms (physical positions [n_q, n_q + 64)), then that
    group is quantized in place with the layout's K/V grouping (V token-wise). Used by the INT4 layouts only."""

    def __init__(self, k_layout: str, bits: int, n_quantized: int, form: str = "min_offset"):
        self.k_layout, self.bits, self.n_q, self.form = k_layout, bits, n_quantized, form
        self.groups_quantized_during_decode = 0
        self.zp_fallback_groups = 0

    @torch.no_grad()
    def __call__(self, cache: DynamicCache):
        P = cache.get_seq_length()
        while P - self.n_q >= KQ.G:
            a, b = self.n_q, self.n_q + KQ.G
            for layer in cache.layers:
                k, v = layer.keys, layer.values
                qk = KQ.quantize_tensor(k[0, :, a:b, :], self.k_layout, self.bits, self.form)
                qv = KQ.quantize_tensor(v[0, :, a:b, :], "token", self.bits, self.form)
                k[0, :, a:b, :], v[0, :, a:b, :] = qk["x_hat"], qv["x_hat"]
                self.zp_fallback_groups += qk["zp_fallback_groups"] + qv["zp_fallback_groups"]
            self.n_q = b
            self.groups_quantized_during_decode += 1


@torch.no_grad()
def run_suffix_and_generate(model, cache, suffix_ids, L, device, max_new: int, eos_ids, after=None) -> dict:
    """Suffix in one call at absolute positions L.., then greedy decoding (stop on EOS, max_new tokens).
    `after(cache)` runs after every forward (INT4 running-group rule)."""
    logits = forward_step(model, cache, suffix_ids, L, device)
    if after:
        after(cache)
    first = logits[-1]
    gen, nxt, pos = [], int(first.argmax()), L + len(suffix_ids)
    finite = bool(torch.isfinite(logits).all())
    while True:
        gen.append(nxt)
        if nxt in eos_ids or len(gen) >= max_new:
            break
        lg = forward_step(model, cache, [nxt], pos, device)
        if after:
            after(cache)
        finite &= bool(torch.isfinite(lg).all())
        pos += 1
        nxt = int(lg[-1].argmax())
    return {"generated": gen, "first_logits": first, "all_finite": finite}


@torch.no_grad()
def teacher_forced_nll(model, cache, suffix_ids, answer_ids, L, device, after=None) -> dict:
    """NLL of the gold answer tokens (code digits + <|im_end|>) after the suffix, token by token (same cache
    semantics as free-running decoding)."""
    lg = forward_step(model, cache, suffix_ids, L, device)
    if after:
        after(cache)
    nll, per = 0.0, []
    pos = L + len(suffix_ids)
    cur = lg[-1]
    for j, t in enumerate(answer_ids):
        lp = torch.log_softmax(cur, -1)
        per.append(float(-lp[t]))
        nll += per[-1]
        if j == len(answer_ids) - 1:
            break
        cur = forward_step(model, cache, [t], pos, device)[-1]
        if after:
            after(cache)
        pos += 1
    return {"nll_sum": nll, "per_token": per, "n_tokens": len(answer_ids)}
