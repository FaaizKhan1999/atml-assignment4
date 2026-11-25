"""Group-wise asymmetric weight quantizer + real bit packer/unpacker (conventions in docs/decisions.md).

Fake-quantized FP16: dequantized weights are evaluated with dense FP16 kernels (no low-bit execution).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

GROUP = 128
S_MIN = 2.0 ** -24  # smallest positive FP16


@dataclass
class QResult:
    codes: torch.Tensor   # [d_out, n_groups, group] int64 in [0, 2^b-1]
    zero: torch.Tensor    # [d_out, n_groups] int64
    scale: torch.Tensor   # [d_out, n_groups] float16 (stored value)
    w_hat: torch.Tensor   # [d_out, d_in] float16 dequantized
    bits: int
    group: int
    d_in: int
    pad: int
    degenerate_groups: int


def dequant(codes, zero, scale, d_in):
    """w_hat = s * (q - z) in FP32 from the FP16 s, cast to FP16; drops padded columns."""
    w = (codes - zero[..., None]).float() * scale.float()[..., None]
    return w.reshape(w.shape[0], -1)[:, :d_in].half()


def quantize(W: torch.Tensor, bits: int, group: int = GROUP, allow_pad: bool = False) -> QResult:
    """PDF quantizer: w_hat = s*(clip(round(w/s)+z, 0, 2^b-1) - z), s=(wmax-wmin)/(2^b-1), z=clamp(round(-wmin/s))."""
    if not 1 <= bits <= 16:
        raise ValueError("bits")
    W32 = W.float()
    d_out, d_in = W32.shape
    pad = (-d_in) % group
    if pad and not allow_pad:
        raise AssertionError(f"d_in={d_in} not divisible by group {group}; padding is disabled")
    Wp = torch.nn.functional.pad(W32, (0, pad))
    valid = torch.ones_like(Wp, dtype=torch.bool)
    if pad:
        valid[:, d_in:] = False
    G = Wp.view(d_out, -1, group)
    V = valid.view(d_out, -1, group)
    wmin = torch.where(V, G, torch.full_like(G, float("inf"))).amin(-1)
    wmax = torch.where(V, G, torch.full_like(G, float("-inf"))).amax(-1)
    qmax = 2 ** bits - 1
    s = ((wmax - wmin) / qmax).clamp(min=S_MIN).half()
    sf = s.float()
    z = torch.clamp(torch.round(-wmin / sf), 0, qmax)
    q = torch.clamp(torch.round(G / sf[..., None]) + z[..., None], 0, qmax)
    q = torch.where(V, q, z[..., None].expand_as(q))  # padded codes = zero-point (dequantize to 0)
    codes, zero = q.long(), z.long()
    return QResult(codes, zero, s, dequant(codes, zero, s, d_in), bits, group, d_in, pad,
                   int((wmax == wmin).sum()))


def bits_per_group(bits: int, group: int = GROUP) -> int:
    return group * bits + bits + 16


def packed_nbytes(d_out: int, d_in: int, bits: int, group: int = GROUP) -> int:
    n_groups = d_out * (-(-d_in // group))
    return -(-(n_groups * bits_per_group(bits, group)) // 8)


def _to_bits(x: torch.Tensor, width: int) -> torch.Tensor:
    sh = torch.arange(width, device=x.device)
    return ((x[..., None] >> sh) & 1).to(torch.uint8).flatten(-2)


def _from_bits(b: torch.Tensor, width: int) -> torch.Tensor:
    sh = torch.arange(width, device=b.device)
    return (b.reshape(*b.shape[:-1], -1, width).long() << sh).sum(-1)


def pack(qr: QResult, chunk_rows: int = 64) -> bytes:
    """One little-endian bitstream per matrix: per (row, group) [zero b][scale 16][codes group*b], byte-padded."""
    if chunk_rows % 8:
        raise ValueError("chunk_rows must be a multiple of 8 so chunk bitstreams are byte aligned")
    d_out = qr.codes.shape[0]
    s_bits = qr.scale.view(torch.int16).long() & 0xFFFF
    out = []
    sh8 = torch.arange(8, device=qr.codes.device)
    for r0 in range(0, d_out, chunk_rows):
        r1 = min(d_out, r0 + chunk_rows)
        f = torch.cat([_to_bits(qr.zero[r0:r1, :, None], qr.bits), _to_bits(s_bits[r0:r1, :, None], 16),
                       _to_bits(qr.codes[r0:r1], qr.bits)], dim=-1).reshape(-1)
        padn = (-f.numel()) % 8
        if padn:
            assert r1 == d_out, "only the final chunk may need padding"
            f = torch.cat([f, f.new_zeros(padn)])
        out.append(((f.view(-1, 8).long() << sh8).sum(-1)).to(torch.uint8).cpu().numpy().tobytes())
    blob = b"".join(out)
    expect = packed_nbytes(d_out, qr.d_in, qr.bits, qr.group)
    if len(blob) != expect:
        raise AssertionError(f"packed {len(blob)} B != formula {expect} B")
    return blob


def unpack(blob: bytes, d_out: int, d_in: int, bits: int, group: int = GROUP, device="cpu",
           chunk_rows: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of pack -> (codes, zero, scale_fp16, w_hat_fp16)."""
    n_g = -(-d_in // group)
    bpg = bits_per_group(bits, group)
    raw = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(device)
    sh8 = torch.arange(8, device=device)
    codes, zeros, scales = [], [], []
    row_bytes_bits = n_g * bpg
    for r0 in range(0, d_out, chunk_rows):
        r1 = min(d_out, r0 + chunk_rows)
        b0 = r0 * row_bytes_bits // 8
        nbits = (r1 - r0) * row_bytes_bits
        seg = raw[b0: b0 + -(-nbits // 8)]
        bitsarr = ((seg[:, None].long() >> sh8) & 1).to(torch.uint8).reshape(-1)[:nbits]
        f = bitsarr.view(r1 - r0, n_g, bpg)
        zeros.append(_from_bits(f[..., :bits], bits).squeeze(-1))
        sv = _from_bits(f[..., bits:bits + 16], 16).squeeze(-1)
        scales.append(torch.where(sv >= 32768, sv - 65536, sv).to(torch.int16).view(torch.float16))
        codes.append(_from_bits(f[..., bits + 16:], bits))
    c, z, s = torch.cat(codes), torch.cat(zeros), torch.cat(scales)
    return c, z, s, dequant(c, z, s, d_in)
