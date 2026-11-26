"""Compressed-cache construction for the Task 2C policies (all at the case-specific budget M_4(L)).

Input: the FP16 context cache Ks/Vs (per layer [1, H_kv, L, d_h], post-RoPE keys kept as they are) and, for the
question-aware rules, one importance score per context position (scoring.py). Output: the tensors loaded into an
ordinary FP16 DynamicCache (dequantized where quantized), the absolute position of every physical entry, and the
ACTUAL serialized byte string (verified by deserializing and comparing bit-exactly with the cache tensors).

Methods
  fp16        : full FP16 reference, all L positions (bytes = M_16(L), no indices)
  sink_recent : C2(i)  8 sinks + 64 recent, then the most recent remaining positions up to n_kept
  qaware      : C2(ii) 8 sinks + 64 recent, then the highest-scoring positions (ties -> lower position) up to n_kept
  int4_tok    : C3(i)  token-wise K and V, INT4, complete 64-token prefix quantized, trailing L mod 64 FP16
  int4_chan   : C3(ii) channel-wise K / token-wise V, INT4, same trailing rule (= the budget reference)
  hybrid      : C4     FP16 set (8 sinks + 64 recent + top-scored extras, f from the exhaustive byte search); the
                complement in sorted position order is INT2 channel-K / token-V in complete 64-token groups; the
                last (n_low mod 64) complement positions stay FP16.
Retention for eviction is ONE global token set shared by all layers and heads.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import kvbytes as KB
from . import kvquant as KQ

METHODS = ("fp16", "sink_recent", "qaware", "int4_tok", "int4_chan", "hybrid")
K_LAYOUT = {"int4_tok": "token", "int4_chan": "channel", "hybrid": "channel"}


@dataclass
class Compressed:
    method: str
    keys: list            # per layer [1, H, P, d] fp16
    values: list
    positions: torch.Tensor  # [P] absolute positions (sorted)
    blob: torch.Tensor     # uint8 serialized bytes
    formula_bytes: int
    breakdown: dict
    fp16_selected: torch.Tensor | None = None  # positions held in FP16 by selection (eviction / hybrid set)
    running: dict | None = None  # INT4 running-group state: {"k_layout", "bits", "n_quantized"}
    info: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return int(self.blob.numel())


def _cat(parts):
    return torch.cat([p.reshape(-1) for p in parts]) if parts else torch.zeros(0, dtype=torch.uint8)


def select_sink_recent(L: int, n: int, device=None) -> torch.Tensor:
    keep = set(range(min(KB.N_SINK, L))) | set(range(max(0, L - (n - KB.N_SINK)), L))
    assert len(keep) == min(n, L)
    return torch.tensor(sorted(keep), device=device)


def select_qaware(scores: torch.Tensor, n: int) -> torch.Tensor:
    L = scores.numel()
    base = set(range(KB.N_SINK)) | set(range(L - KB.N_RECENT, L))
    rest = [i for i in range(L) if i not in base]
    s = scores.detach().float().cpu()
    rest_sorted = sorted(rest, key=lambda i: (-float(s[i]), i))
    keep = sorted(base | set(rest_sorted[:max(0, n - len(base))]))
    assert len(keep) == min(n, L)
    return torch.tensor(keep, device=scores.device)


def _u16(pos: torch.Tensor) -> torch.Tensor:
    assert int(pos.max()) < 65536
    return pos.to(torch.int32).to(torch.int16).contiguous().view(torch.uint8)  # little-endian uint16 bit pattern


def compress(method: str, Ks: list, Vs: list, cfg, scores: torch.Tensor | None = None) -> Compressed:
    L = Ks[0].shape[2]
    dev = Ks[0].device
    assert Ks[0].dtype == torch.float16 and Ks[0].shape[0] == 1, "FP16 batch-1 cache expected"
    C = KB.channels(cfg)
    budget = KB.m_b(L, C, 4)
    allpos = torch.arange(L, device=dev)
    if method == "fp16":
        parts = [p for k, v in zip(Ks, Vs) for p in (KQ._u8(k), KQ._u8(v))]
        return Compressed(method, [k.clone() for k in Ks], [v.clone() for v in Vs], allpos, _cat(parts),
                          KB.m16(L, C), {"fp16_bytes": KB.m16(L, C)}, fp16_selected=allpos)
    if method in ("sink_recent", "qaware"):
        n = KB.n_kept_max(budget, C, L)
        pos = select_sink_recent(L, n, dev) if method == "sink_recent" else select_qaware(scores, n)
        keys = [k[:, :, pos, :].contiguous() for k in Ks]
        vals = [v[:, :, pos, :].contiguous() for v in Vs]
        parts = [p for k, v in zip(keys, vals) for p in (KQ._u8(k), KQ._u8(v))] + [_u16(pos)]
        fb = KB.eviction_bytes(n, C)
        return Compressed(method, keys, vals, pos, _cat(parts), fb,
                          {"fp16_bytes": fb - KB.IDX * n, "index_bytes": KB.IDX * n, "n_kept": n, "budget": budget},
                          fp16_selected=pos)
    if method in ("int4_tok", "int4_chan"):
        nq = (L // KB.G) * KB.G
        kl = K_LAYOUT[method]
        keys, vals, parts = [], [], []
        for k, v in zip(Ks, Vs):
            qk = KQ.quantize_tensor(k[0, :, :nq, :], kl, 4)
            qv = KQ.quantize_tensor(v[0, :, :nq, :], "token", 4)
            keys.append(torch.cat([qk["x_hat"], k[0, :, nq:, :]], 1)[None])
            vals.append(torch.cat([qv["x_hat"], v[0, :, nq:, :]], 1)[None])
            parts += KQ.serialize_quant(qk) + KQ.serialize_quant(qv) + [KQ._u8(k[0, :, nq:, :]), KQ._u8(v[0, :, nq:, :])]
        d = Ks[0].shape[-1]
        fb = KB.m_b(L, C, 4) if kl == "channel" else KB.tokenwise_bytes(L, C, d, 4)
        assert fb == budget, "token-wise and channel-wise INT4 bytes must coincide (64 | d_h)"
        return Compressed(method, keys, vals, allpos, _cat(parts), fb,
                          {"quantized_tokens": nq, "residual_fp16_tokens": L - nq, "budget": budget},
                          running={"k_layout": kl, "bits": 4, "n_quantized": nq})
    if method == "hybrid":
        f = KB.hybrid_f_max(budget, L, C, 2)
        sel = select_qaware(scores, f)
        mask = torch.zeros(L, dtype=torch.bool, device=dev)
        mask[sel] = True
        low = allpos[~mask]
        hb = KB.hybrid_bytes(f, L, C, 2)
        g = hb["low_groups_per_channel"]
        qpos = low[:g * KB.G]
        resid = low[g * KB.G:]
        fp_pos = torch.sort(torch.cat([sel, resid])).values
        keys, vals, parts = [], [], []
        for k, v in zip(Ks, Vs):
            kk, vv = k[0].clone(), v[0].clone()
            fk, fv = k[0, :, fp_pos, :], v[0, :, fp_pos, :]
            if g:
                qk = KQ.quantize_tensor(k[0, :, qpos, :], "channel", 2)
                qv = KQ.quantize_tensor(v[0, :, qpos, :], "token", 2)
                kk[:, qpos, :] = qk["x_hat"]
                vv[:, qpos, :] = qv["x_hat"]
                parts += KQ.serialize_quant(qk) + KQ.serialize_quant(qv)
            parts += [KQ._u8(fk), KQ._u8(fv)]
            keys.append(kk[None])
            vals.append(vv[None])
        parts.append(_u16(sel))
        return Compressed(method, keys, vals, allpos, _cat(parts), hb["total"], {**hb, "budget": budget},
                          fp16_selected=sel, info={"low_quantized_positions": int(qpos.numel()),
                                                   "residual_positions": resid.tolist()})
    raise ValueError(method)


def deserialize(c: Compressed, cfg, L: int) -> tuple[list, list]:
    """Rebuild the physical FP16 cache tensors from the serialized bytes alone (+ L and the method)."""
    C = KB.channels(cfg)
    H, d = cfg.num_key_value_heads, KB.channels(cfg) // (cfg.num_hidden_layers * cfg.num_key_value_heads)
    buf, off = c.blob, 0
    nl = cfg.num_hidden_layers
    keys, vals = [], []

    def take_fp16(n):
        nonlocal off
        t = buf[off:off + 2 * H * n * d].view(torch.float16).view(H, n, d)
        off += 2 * H * n * d
        return t

    if c.method == "fp16":
        for _ in range(nl):
            keys.append(take_fp16(L)[None])
            vals.append(take_fp16(L)[None])
    elif c.method in ("sink_recent", "qaware"):
        n = KB.n_kept_max(KB.m_b(L, C, 4), C, L)
        for _ in range(nl):
            keys.append(take_fp16(n)[None])
            vals.append(take_fp16(n)[None])
        pos = buf[off:off + 2 * n].view(torch.int16).to(torch.int64) & 0xFFFF
        off += 2 * n
        assert torch.equal(pos, c.positions.to(pos.device))
    elif c.method in ("int4_tok", "int4_chan"):
        nq = (L // KB.G) * KB.G
        for _ in range(nl):
            k, off = KQ.deserialize_quant(buf, off, (H, nq, d), K_LAYOUT[c.method], 4)
            v, off = KQ.deserialize_quant(buf, off, (H, nq, d), "token", 4)
            keys.append(torch.cat([k, take_fp16(L - nq)], 1)[None])
            vals.append(torch.cat([v, take_fp16(L - nq)], 1)[None])
    elif c.method == "hybrid":
        f = c.breakdown["f_selected"]
        g = c.breakdown["low_groups_per_channel"]
        r = c.breakdown["residual_fp16"]
        layer_parts = []
        for _ in range(nl):
            k = v = None
            if g:
                k, off = KQ.deserialize_quant(buf, off, (H, g * KB.G, d), "channel", 2)
                v, off = KQ.deserialize_quant(buf, off, (H, g * KB.G, d), "token", 2)
            layer_parts.append((k, v, take_fp16(f + r), take_fp16(f + r)))
        sel = buf[off:off + 2 * f].view(torch.int16).to(torch.int64) & 0xFFFF
        off += 2 * f
        mask = torch.zeros(L, dtype=torch.bool, device=buf.device)
        mask[sel] = True
        low = torch.arange(L, device=buf.device)[~mask]
        qpos, resid = low[:g * KB.G], low[g * KB.G:]
        fp_pos = torch.sort(torch.cat([sel, resid])).values
        for k, v, fk, fv in layer_parts:
            K = torch.empty(H, L, d, dtype=torch.float16, device=buf.device)
            V = torch.empty_like(K)
            if g:
                K[:, qpos], V[:, qpos] = k, v
            K[:, fp_pos], V[:, fp_pos] = fk, fv
            keys.append(K[None])
            vals.append(V[None])
    else:
        raise ValueError(c.method)
    assert off == buf.numel(), f"trailing bytes {buf.numel() - off}"
    return keys, vals


def reconstruction_error(c: Compressed, Ks: list, Vs: list) -> dict | None:
    """Relative Frobenius error of K and V separately over all L context positions (quantized layouts, hybrid)."""
    if c.method not in ("int4_tok", "int4_chan", "hybrid"):
        return None
    out = {}
    for name, ref, rec in (("K", Ks, c.keys), ("V", Vs, c.values)):
        per = [float(torch.linalg.vector_norm((a.float() - b.float())) / torch.linalg.vector_norm(b.float()))
               for a, b in zip(rec, ref)]
        out[name] = {"per_layer": per, "mean": sum(per) / len(per)}
    return out
