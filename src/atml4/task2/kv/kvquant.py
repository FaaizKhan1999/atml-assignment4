"""KV group quantization (asymmetric, group 64), bit packing and exact serialization.

Quantizer (asymmetric min/scale form, b in {2, 4}, group 64). Metadata per group = FP16 scale s + FP16 offset o
(4 bytes). The metadata are rounded to FP16 FIRST and the integer codes are computed against the STORED values:
  o = largest FP16 value <= x_min (round toward -inf);  s = smallest FP16 value >= max((x_max - o)/(2^b - 1), 2^-24),
  bumped to the next FP16 value while o + (2^b - 1) s < x_max (so the stored grid covers the group's range);
  q = clamp(round((x - o)/s), 0, 2^b - 1)  (round half to even);  x_hat = o + q s  (FP32 from the stored FP16 values,
  then cast to FP16 for ordinary SDPA).
(Fix 2026-11-26, docs/deviations.md: the first version reused Task 1's zero-point form z = clamp(round(-x_min/s),
0, 2^b - 1). For groups that do not contain 0 -- e.g. a key channel with a large bias over 64 tokens -- z was clamped
and the whole group collapsed. Task 1's weight quantizer is unchanged.)
Dequantized values are FP16 tensors in an ordinary DynamicCache ("dequantized to FP16 for SDPA"): packed storage
!= allocated GPU memory.

Layouts on one layer's K or V tensor X [H_kv, N, d_h] (N tokens, multiple of 64 where grouped along tokens):
  channel : groups of 64 consecutive (stored-order) tokens per (head, channel)   -> X.view(H, N/64, 64, d) -> (H, N/64, d, 64)
  token   : groups of 64 consecutive features per (head, token)                 -> X.view(H, N, d/64, 64)
Serialization (no header; shapes are implied by config + L + method): per layer [K payload | K scales | K offsets |
V payload | V scales | V offsets | K FP16 part | V FP16 part], then uint16 position indices (if any). Codes are packed
little-endian within a byte (code j of a byte at bit offset j*b) in the flattened group order above.
"""
from __future__ import annotations

import torch

G = 64
S_MIN = 2.0 ** -24


def _groups(X: torch.Tensor, layout: str) -> torch.Tensor:
    H, N, d = X.shape
    if layout == "channel":
        assert N % G == 0
        return X.reshape(H, N // G, G, d).permute(0, 1, 3, 2)  # (H, N/64, d, 64)
    if layout == "token":
        assert d % G == 0
        return X.reshape(H, N, d // G, G)
    raise ValueError(layout)


def _ungroup(Y: torch.Tensor, layout: str, shape) -> torch.Tensor:
    H, N, d = shape
    if layout == "channel":
        return Y.permute(0, 1, 3, 2).reshape(H, N, d)
    return Y.reshape(H, N, d)


def _fp16_down(x: torch.Tensor) -> torch.Tensor:
    """Largest FP16 value <= x (x FP32)."""
    h = x.half()
    up = h.float() > x
    return torch.where(up, torch.nextafter(h, torch.full_like(h, float("-inf"))), h)


def _fp16_up(x: torch.Tensor) -> torch.Tensor:
    """Smallest FP16 value >= x (x FP32)."""
    h = x.half()
    dn = h.float() < x
    return torch.where(dn, torch.nextafter(h, torch.full_like(h, float("inf"))), h)


def quantize_groups(Gx: torch.Tensor, bits: int):
    """Gx [..., 64] -> codes uint8 [..., 64], scale fp16 [...], offset fp16 [...], x_hat fp32 [..., 64]."""
    x = Gx.float()
    qmax = 2 ** bits - 1
    xmin, xmax = x.amin(-1), x.amax(-1)
    o = _fp16_down(xmin)
    of = o.float()
    s = _fp16_up(((xmax - of) / qmax).clamp(min=S_MIN))
    for _ in range(4):  # FP32 rounding of o + qmax*s: guarantee coverage of x_max with the stored values
        short = (of + qmax * s.float()) < xmax
        if not bool(short.any()):
            break
        s = torch.where(short, torch.nextafter(s, torch.full_like(s, float("inf"))), s)
    sf = s.float()
    q = torch.clamp(torch.round((x - of[..., None]) / sf[..., None]), 0, qmax)
    xh = of[..., None] + q * sf[..., None]
    return q.to(torch.uint8), s, o, xh


def dequant_groups(codes: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    return offset.float()[..., None] + codes.float() * scale.float()[..., None]


def quantize_tensor(X: torch.Tensor, layout: str, bits: int) -> dict:
    """X [H, N, d] -> {codes, scale, offset, x_hat (fp16, X shape)}."""
    c, s, o, xh = quantize_groups(_groups(X, layout), bits)
    return {"codes": c, "scale": s, "offset": o, "x_hat": _ungroup(xh, layout, X.shape).half(), "layout": layout,
            "bits": bits, "shape": tuple(X.shape)}


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    per = 8 // bits
    flat = codes.reshape(-1).to(torch.uint8)
    assert flat.numel() % per == 0
    flat = flat.view(-1, per).to(torch.int32)
    out = torch.zeros(flat.shape[0], dtype=torch.int32, device=codes.device)
    for j in range(per):
        out |= flat[:, j] << (j * bits)
    return out.to(torch.uint8)


def unpack(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    per = 8 // bits
    p = packed.to(torch.int32)
    mask = (1 << bits) - 1
    out = torch.stack([(p >> (j * bits)) & mask for j in range(per)], dim=1).reshape(-1)[:n]
    return out.to(torch.uint8)


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(-1)


def serialize_quant(q: dict) -> list[torch.Tensor]:
    return [pack(q["codes"], q["bits"]), _u8(q["scale"]), _u8(q["offset"])]


def deserialize_quant(buf: torch.Tensor, off: int, shape, layout: str, bits: int) -> tuple[torch.Tensor, int]:
    H, N, d = shape
    ng = (H * (N // G) * d) if layout == "channel" else (H * N * (d // G))
    nb = ng * G * bits // 8
    codes = unpack(buf[off:off + nb], bits, ng * G)
    off += nb
    scale = buf[off:off + 2 * ng].view(torch.float16)
    off += 2 * ng
    offset = buf[off:off + 2 * ng].view(torch.float16)
    off += 2 * ng
    gshape = (H, N // G, d, G) if layout == "channel" else (H, N, d // G, G)
    xh = dequant_groups(codes.view(gshape), scale.view(gshape[:-1]), offset.view(gshape[:-1]))
    return _ungroup(xh, layout, shape).half(), off
