"""Task 2 Part B: small decoder-only Transformer for MQAR with a pluggable attention operator.

2 pre-norm blocks, d_model 128, 4 heads x 32, MLP 4x (GELU, exact erf form), LayerNorm (eps 1e-5), learned token
embeddings (vocab 256), fixed sinusoidal positional embeddings (Vaswani et al.), final LayerNorm, and a 64-way
linear head over value IDs 128..191 applied only at the marked query positions. Only the attention operator
differs between architectures:
  dense   : causal softmax attention (SDPA, is_causal=True)
  window  : causal sliding window W=64, j in [i-63, i]   (EXPLICIT DENSE BOOLEAN MASK; correctness implementation)
  dilated : heads 0-1 local [i-63, i]; heads 2-3 j = i - o, o in {0,4,...,252}, j >= 0 (EXPLICIT DENSE MASK)
  linear  : causal ELU+1 linear attention; training uses the cumulative-sum form, which MATERIALISES prefix
            tensors S_t of shape [B,H,T,32,32]; `recurrent=True` uses the O(1)-state recurrence.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 256
D_MODEL = 128
N_HEADS = 4
D_HEAD = 32
N_LAYERS = 2
WINDOW = 64
DILATION = 4
N_DILATED = 64
VALUE_LO, N_VALUES = 128, 64
LINEAR_EPS = 1e-6
ARCHS = ("dense", "window", "dilated", "linear")


def window_mask(T: int, W: int = WINDOW, device=None) -> torch.Tensor:
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (j > i - W)


def dilated_mask(T: int, device=None) -> torch.Tensor:
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    off = i - j
    return (off >= 0) & (off % DILATION == 0) & (off <= DILATION * (N_DILATED - 1))


def head_masks(arch: str, T: int, device=None) -> torch.Tensor | None:
    """[H, T, T] boolean (True = attend) or None for dense/linear."""
    if arch == "window":
        return window_mask(T, device=device)[None].expand(N_HEADS, T, T)
    if arch == "dilated":
        w, d = window_mask(T, device=device), dilated_mask(T, device=device)
        return torch.stack([w, w, d, d])
    return None


def phi(x):
    return F.elu(x) + 1.0


def linear_attention_cumsum(q, k, v, eps=LINEAR_EPS):
    """[B,H,T,d] -> causal ELU+1 linear attention via cumulative sums (materialises [B,H,T,d,d] prefixes)."""
    fq, fk = phi(q), phi(k)
    S = torch.cumsum(fk[..., :, None] * v[..., None, :], dim=2)      # S_t = sum_{s<=t} phi(k_s) v_s^T
    z = torch.cumsum(fk, dim=2)                                         # z_t = sum_{s<=t} phi(k_s)
    num = (fq[..., None, :] @ S).squeeze(-2)                            # phi(q_t)^T S_t
    den = (fq * z).sum(-1, keepdim=True) + eps                          # phi(q_t)^T z_t + eps
    return num / den


def linear_attention_recurrent(q, k, v, eps=LINEAR_EPS):
    """Same function with the O(1)-state recurrence S_t = S_{t-1} + phi(k_t) v_t^T, z_t = z_{t-1} + phi(k_t)."""
    B, H, T, d = q.shape
    S = q.new_zeros(B, H, d, v.shape[-1])
    z = q.new_zeros(B, H, d)
    out = []
    fq, fk = phi(q), phi(k)
    for t in range(T):
        S = S + fk[:, :, t, :, None] * v[:, :, t, None, :]
        z = z + fk[:, :, t]
        num = (fq[:, :, t, None, :] @ S).squeeze(-2)
        den = (fq[:, :, t] * z).sum(-1, keepdim=True) + eps
        out.append(num / den)
    return torch.stack(out, 2)


def sinusoidal(T: int, d: int, device=None) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe


class Attention(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        self.arch = arch
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mask_override = None  # e.g. "window" for the train/test mask control
        self.recurrent = False

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, N_HEADS, D_HEAD).permute(2, 0, 3, 1, 4)
        arch = self.mask_override or self.arch
        if arch == "linear":
            o = (linear_attention_recurrent if self.recurrent else linear_attention_cumsum)(q, k, v)
        elif arch == "dense":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            m = head_masks(arch, T, x.device)[None]
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.out(o.transpose(1, 2).reshape(B, T, D_MODEL))


class Block(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attention(arch)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MQARModel(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(arch)
        self.arch = arch
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block(arch) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_VALUES)

    def set_mask_override(self, arch: str | None):
        for b in self.blocks:
            b.attn.mask_override = arch

    def forward(self, tokens, query_pos):
        """tokens [B,T]; query_pos [B,K] -> logits [B,K,64] at the marked query positions only."""
        T = tokens.shape[1]
        x = self.emb(tokens) + sinusoidal(T, D_MODEL, tokens.device)[None]
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        h = x.gather(1, query_pos[..., None].expand(-1, -1, D_MODEL))
        return self.head(h)


def query_loss(logits, answers):
    """Cross-entropy at query positions only: answers are value IDs (128..191) -> class = id - 128."""
    return F.cross_entropy(logits.reshape(-1, N_VALUES), (answers - VALUE_LO).reshape(-1))


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lr_at(update: int, peak=8e-4, warmup=200, total=4500, final_frac=0.15) -> float:
    """update is the 0-based optimizer-update index: linear warmup peak*(u+1)/warmup for u < warmup, then linear
    decay from peak at u = warmup to final_frac*peak at u = total."""
    if update < warmup:
        return peak * (update + 1) / warmup
    return peak * (1 - (1 - final_frac) * (update - warmup) / (total - warmup))


def theoretical_access(T: int = 256) -> dict:
    """Attended (query,key) pairs per layer per head, and per-query reachable positions."""
    dense = T * (T + 1) // 2
    win = sum(min(i + 1, WINDOW) for i in range(T))
    dil = sum(min(N_DILATED, i // DILATION + 1) for i in range(T))
    return {"T": T, "pairs_total_per_head": {"dense": dense, "window": win, "dilated_heads": dil,
                                             "mixed_local_dilated_mean_over_heads": (2 * win + 2 * dil) / 4,
                                             "linear": "n/a (state of size d x d + d per head)"},
            "max_positions_per_query": {"dense": T, "window": WINDOW, "dilated_heads": N_DILATED,
                                        "linear": "all (compressed into fixed-size state)"},
            "receptive_field_two_layers": {"window": 2 * (WINDOW - 1),
                                           "dilated_heads": 2 * DILATION * (N_DILATED - 1)}}


def state_bytes(T: int = 256, elem_bytes: int = 4) -> dict:
    """Persistent decode-time state per layer (all heads): KV cache vs linear S,z."""
    kv = lambda n: 2 * n * N_HEADS * D_HEAD * elem_bytes  # noqa: E731
    return {"dense_kv": kv(T), "window_kv": kv(min(T, WINDOW)),
            "dilated_mixed_kv": 2 * (2 * min(T, WINDOW) + 2 * min(T, DILATION * (N_DILATED - 1) + 1)) * D_HEAD * elem_bytes,
            "linear_S_z": N_HEADS * (D_HEAD * D_HEAD + D_HEAD) * elem_bytes, "elem_bytes": elem_bytes, "T": T}
