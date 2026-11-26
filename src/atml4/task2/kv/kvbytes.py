"""Exact KV-cache byte accounting (pure functions; LOGICAL/serialized bytes, never GPU-allocated bytes).

Notation: C = n_l * H_kv * d_h = stored channels per K (and per V); G = 64 (group size); FP16 = 2 bytes.
* M_b(L) = 2C [ floor(L/64) (64 b / 8 + 4) + 2 (L mod 64) ]   (PDF; channel-wise K / token-wise V, b-bit)
* M_16(L) = 4 C L
* Token-wise K and V: per stored token, d_h/64 groups per head -> same element count, same metadata, same FP16
  trailing rule -> identical bytes when 64 | d_h (asserted).
* FP16 eviction (ONE global retained token set shared by all layers and heads): 4 C n + 2 n (uint16 indices,
  valid because L < 65,536, stored once).
* Hybrid: FP16 set of size f (8 sinks + 64 recent + extras, uint16 indices stored once); the complement (n_low =
  L - f positions, sorted) is INT2 channel-K / token-V in complete 64-token groups; the incomplete residual
  (n_low mod 64 positions, the LAST ones of the sorted complement) is FP16 and needs no index (it is derivable).
  bytes = 2C floor(n_low/64) (64*2/8 + 4) + 4C (f + n_low mod 64) + 2 f.
"""
from __future__ import annotations

G = 64
FP16 = 2
IDX = 2  # uint16
N_SINK, N_RECENT = 8, 64


def channels(cfg) -> int:
    """n_l * H_kv * d_h read from the model config (no hard-coding)."""
    d_h = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return cfg.num_hidden_layers * cfg.num_key_value_heads * d_h


def group_bytes(bits: int) -> int:
    assert (G * bits) % 8 == 0
    return G * bits // 8 + 4  # payload + FP16 scale + FP16 offset


def m_b(L: int, C: int, bits: int = 4) -> int:
    return 2 * C * ((L // G) * group_bytes(bits) + FP16 * (L % G))


def m16(L: int, C: int) -> int:
    return 2 * FP16 * C * L


def tokenwise_bytes(L: int, C: int, d_h: int, bits: int = 4) -> int:
    """Token-wise K and V: quantized tokens = complete 64-token prefix part (same trailing rule); per token and
    head, d_h/64 feature groups."""
    assert d_h % G == 0
    n_q, r = (L // G) * G, L % G
    per_token_groups = (C // d_h) * (d_h // G)  # n_l * H_kv * d_h/64
    return 2 * (n_q * per_token_groups * group_bytes(bits)) + 2 * FP16 * C * r


def eviction_bytes(n: int, C: int) -> int:
    return 2 * FP16 * C * n + IDX * n


def n_kept_max(budget: int, C: int, L: int) -> int:
    n = min(L, budget // (2 * FP16 * C + IDX))
    if n < N_SINK + N_RECENT:
        raise ValueError("budget cannot hold the 8 sinks + 64 recent safeguards")
    assert eviction_bytes(n, C) <= budget and (n == L or eviction_bytes(n + 1, C) > budget)
    return n


def hybrid_bytes(f: int, L: int, C: int, bits_low: int = 2) -> dict:
    n_low = L - f
    g, r = n_low // G, n_low % G
    low = 2 * C * g * group_bytes(bits_low)
    fp = 2 * FP16 * C * (f + r)
    idx = IDX * f
    return {"f_selected": f, "n_low": n_low, "low_groups_per_channel": g, "residual_fp16": r, "low_bytes": low,
            "low_payload_bytes": 2 * C * g * (G * bits_low // 8), "low_metadata_bytes": 2 * C * g * 4,
            "fp16_bytes": fp, "index_bytes": idx, "total": low + fp + idx}


def hybrid_f_max(budget: int, L: int, C: int, bits_low: int = 2) -> int:
    """Largest FP16 selected-set size f in [72, L] with hybrid_bytes(f) <= budget, by exhaustive search (the
    residual makes the cost non-monotone in f)."""
    best = None
    for f in range(N_SINK + N_RECENT, L + 1):
        if hybrid_bytes(f, L, C, bits_low)["total"] <= budget:
            best = f
    if best is None:
        raise ValueError("hybrid infeasible at this budget")
    return best


def live_quantized_bytes(n_quantized_tokens: int, n_fp16_tokens: int, C: int, bits: int = 4) -> int:
    """Running INT4 layout: complete groups of quantized tokens + FP16 tokens (residual / not yet grouped)."""
    assert n_quantized_tokens % G == 0
    return 2 * C * (n_quantized_tokens // G) * group_bytes(bits) + 2 * FP16 * C * n_fp16_tokens
