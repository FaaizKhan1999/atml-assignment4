"""KV group quantization (asymmetric, group 64), bit packing and exact serialization.

Quantizer = Task 1's asymmetric form (atml4.task1.quant.quantize) with b in {2, 4} and group 64:
  s = max((x_max - x_min) / (2^b - 1), 2^-24) rounded to FP16;  z = clamp(round(-x_min / s), 0, 2^b - 1);
  q = clamp(round(x / s) + z, 0, 2^b - 1);  x_hat = s (q - z)   (FP32 arithmetic, round half to even, x_hat cast
  to FP16 for ordinary SDPA). Metadata per group = FP16 scale + FP16 offset, where the offset field stores the
  integer zero-point z as an FP16 value (exact for z <= 15). Dequantized values are FP16 tensors in an ordinary
  DynamicCache ("dequantized to FP16 for SDPA"): packed storage != allocated GPU memory.

Layouts on one layer's K or V tensor X [H_kv, N, d_h] (N tokens, multiple of 64 where grouped along tokens):
  channel : groups of 64 consecutive (stored-order) tokens per (head, channel)   -> X.view(H, N/64, 64, d) -> (H, N/64, d, 64)
  token   : groups of 64 consecutive features per (head, token)                 -> X.view(H, N, d/64, 64)
Serialization (no header; shapes are implied by config + L + method): per layer [K payload | K scales | K zeros |
V payload | V scales | V zeros | K FP16 part | V FP16 part], then uint16 position indices (if any). Codes are packed
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


def quantize_groups(Gx: torch.Tensor, bits: int):
    """Gx [..., 64] -> codes uint8 [..., 64], scale fp16 [...], zero fp16 [...], x_hat fp32 [..., 64]."""
    x = Gx.float()
    qmax = 2 ** bits - 1
    xmin, xmax = x.amin(-1), x.amax(-1)
    s = ((xmax - xmin) / qmax).clamp(min=S_MIN).half()
    sf = s.float()
    z = torch.clamp(torch.round(-xmin / sf), 0, qmax)
    q = torch.clamp(torch.round(x / sf[..., None]) + z[..., None], 0, qmax)
    xh = (q - z[..., None]) * sf[..., None]
    return q.to(torch.uint8), s, z.half(), xh


def dequant_groups(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    return (codes.float() - zero.float()[..., None]) * scale.float()[..., None]


def quantize_tensor(X: torch.Tensor, layout: str, bits: int) -> dict:
    """X [H, N, d] -> {'codes','scale','zero','x_hat' (fp16, X's shape)}."""
    c, s, z, xh = quantize_groups(_groups(X, layout), bits)
    return {"codes": c, "scale": s, "zero": z, "x_hat": _ungroup(xh, layout, X.shape).half(), "layout": layout,
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
    return [pack(q["codes"], q["bits"]), _u8(q["scale"]), _u8(q["zero"])]


def deserialize_quant(buf: torch.Tensor, off: int, shape, layout: str, bits: int) -> tuple[torch.Tensor, int]:
    H, N, d = shape
    ng = (H * (N // G) * d) if layout == "channel" else (H * N * (d // G))
    nb = ng * G * bits // 8
    codes = unpack(buf[off:off + nb], bits, ng * G)
    off += nb
    scale = buf[off:off + 2 * ng].view(torch.float16)
    off += 2 * ng
    zero = buf[off:off + 2 * ng].view(torch.float16)
    off += 2 * ng
    gshape = (H, N // G, d, G) if layout == "channel" else (H, N, d // G, G)
    xh = dequant_groups(codes.view(gshape), scale.view(gshape[:-1]), zero.view(gshape[:-1]))
    return _ungroup(xh, layout, shape).half(), off
