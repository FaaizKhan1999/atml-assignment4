"""Task 2 Part A attention operators: dense causal (explicit), tiled online-softmax, PyTorch SDPA.

Conventions (docs/decisions.md):
* scale 1/sqrt(d); causal: query i attends keys j <= i.
* Fully-masked rows (causal AND key-padding mask exclude every key) return 0 and receive zero gradients.
* Dense computes scores/softmax/PV in the INPUT dtype (FP16 stays FP16) with max-subtraction.
* Tiled: query and key tiles of size `tile`; m, l, o accumulators in FP32 (scores upcast); output cast back.
  The L x L matrix is never materialised, but autograd through the Python loop STORES per-tile intermediates
  (this is a correctness reference, not FlashAttention).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def causal_mask(Lq: int, Lk: int, device) -> torch.Tensor:
    """True = allowed (j <= i)."""
    return torch.ones(Lq, Lk, dtype=torch.bool, device=device).tril()


def dense_attention(q, k, v, key_padding_mask=None, stable: bool = True):
    """Explicit QK^T, causal (+optional key padding; True = keep) mask, softmax, PV. [B,H,L,d] tensors."""
    d = q.shape[-1]
    s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(d))
    allowed = causal_mask(q.shape[-2], k.shape[-2], q.device)[None, None]
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask[:, None, None, :]
    neg = torch.tensor(float("-inf"), dtype=s.dtype, device=s.device)
    s = torch.where(allowed, s, neg)
    row_ok = allowed.any(-1, keepdim=True)
    if stable:
        m = s.amax(-1, keepdim=True)
        m = torch.where(row_ok, m, torch.zeros_like(m)).detach()
        e = torch.exp(s - m)
    else:
        e = torch.exp(s)
    denom = e.sum(-1, keepdim=True)
    p = e / torch.where(row_ok, denom, torch.ones_like(denom))
    return torch.where(row_ok, p @ v, torch.zeros((), dtype=v.dtype, device=v.device))


def dense_reference_fp32(q, k, v, grad_out=None, chunk: int = 1024):
    """Exact dense causal attention in FP32 computed in query chunks (each chunk materialises only
    [chunk x L] scores). Returns (out, (dq, dk, dv) or None). Used as the reference at every L."""
    q32, k32, v32 = (t.detach().float().requires_grad_(grad_out is not None) for t in (q, k, v))
    L = q.shape[-2]
    d = q.shape[-1]
    outs = []
    for i0 in range(0, L, chunk):
        i1 = min(L, i0 + chunk)
        s = (q32[..., i0:i1, :] @ k32[..., :i1, :].transpose(-1, -2)) / math.sqrt(d)
        allowed = torch.ones(i1 - i0, i1, dtype=torch.bool, device=q.device).tril(diagonal=i0)
        s = s.masked_fill(~allowed, float("-inf"))
        o = torch.softmax(s, -1) @ v32[..., :i1, :]
        if grad_out is not None:
            (o * grad_out[..., i0:i1, :].float()).sum().backward()
        outs.append(o.detach())
    out = torch.cat(outs, -2)
    grads = (q32.grad, k32.grad, v32.grad) if grad_out is not None else None
    return out, grads


def tiled_attention(q, k, v, tile: int = 128):
    """Exact causal attention with the online-softmax update (PDF equations), query/key tiles of size `tile`."""
    B, H, L, d = q.shape
    scale = 1.0 / math.sqrt(d)
    outs = []
    for i0 in range(0, L, tile):
        i1 = min(L, i0 + tile)
        qi = q[..., i0:i1, :]
        m = torch.full((B, H, i1 - i0, 1), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, i1 - i0, 1), device=q.device, dtype=torch.float32)
        o = torch.zeros((B, H, i1 - i0, d), device=q.device, dtype=torch.float32)
        for j0 in range(0, i1, tile):
            j1 = min(L, j0 + tile)
            s = (qi @ k[..., j0:j1, :].transpose(-1, -2)).float() * scale
            if j1 > i0:  # diagonal tile: causal mask inside the tile
                allowed = (torch.arange(i0, i1, device=q.device)[:, None] >=
                           torch.arange(j0, j1, device=q.device)[None, :])
                s = s.masked_fill(~allowed, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            alpha = torch.exp(m - m_new)
            p = torch.exp(s - m_new)
            l = alpha * l + p.sum(-1, keepdim=True)
            o = alpha * o + p @ v[..., j0:j1, :].float()
            m = m_new
        outs.append((o / l).to(q.dtype))
    return torch.cat(outs, -2)


def tiled_intermediate_bytes(B, H, L, d, tile, elem_bytes=4) -> dict:
    """Analytic storage: one live score tile vs the full L x L score matrix (per forward, excl. autograd)."""
    n_tiles = sum(range(1, -(-L // tile) + 1))
    return {"live_score_tile_bytes": B * H * tile * tile * elem_bytes,
            "accumulators_bytes": B * H * tile * (d + 2) * elem_bytes,
            "dense_score_matrix_bytes": B * H * L * L * elem_bytes,
            "tiles_processed": n_tiles,
            "autograd_saved_score_tiles_bytes_approx": n_tiles * B * H * tile * tile * elem_bytes}


def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
