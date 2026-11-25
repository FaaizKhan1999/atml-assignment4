"""AWQ-style activation-aware per-channel scale search (Q2) on top of the shared quantizer (conventions: decisions.md).

Reproduced: per-channel scales s = mean|x|^alpha (llm-awq normalisation), grid search per scale group on
calibration activations from the UNCHANGED FP16 model, folding down_proj scales into up_proj rows.
Not reproduced: AWQ clipping search, fused INT4 kernels / inference engine, module-output objective.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from .quant import quantize


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


ALPHAS = [i / 20 for i in range(20)]
GROUPS = [("qkv", ("q_proj", "k_proj", "v_proj")), ("o", ("o_proj",)), ("gateup", ("gate_proj", "up_proj")),
          ("down", ("down_proj",))]
SERIALIZED = ("qkv", "o", "gateup")  # preceded by RMSNorm or GQA-incompatible v_proj -> FP16 scales serialized
FOLDED = ("down",)                    # folded into up_proj output rows


def _mods(layer):
    a, m = layer.self_attn, layer.mlp
    return {"q_proj": a.q_proj, "k_proj": a.k_proj, "v_proj": a.v_proj, "o_proj": a.o_proj,
            "gate_proj": m.gate_proj, "up_proj": m.up_proj, "down_proj": m.down_proj}


def check_fold_validity(model) -> dict:
    layer = model.model.layers[0]
    v, o = layer.self_attn.v_proj, layer.self_attn.o_proj
    up, down = layer.mlp.up_proj, layer.mlp.down_proj
    return {"v_proj_out": v.weight.shape[0], "o_proj_in": o.weight.shape[1],
            "o_from_v_fold_valid": v.weight.shape[0] == o.weight.shape[1],
            "up_proj_out": up.weight.shape[0], "down_proj_in": down.weight.shape[1],
            "down_into_up_fold_valid": up.weight.shape[0] == down.weight.shape[1] and up.bias is None}


class _Stop(Exception):
    pass


@torch.no_grad()
def capture_block_inputs(model, li: int, batches) -> dict[str, torch.Tensor]:
    """Inputs of the four scale groups of block li, from the unchanged model (forward stops after block li)."""
    layer = model.model.layers[li]
    mods = _mods(layer)
    caps = {g: [] for g, _ in GROUPS}
    hooks = []
    for g, names in GROUPS:
        hooks.append(mods[names[0]].register_forward_pre_hook(
            lambda mod, args, g=g: caps[g].append(args[0].detach().reshape(-1, args[0].shape[-1]))))

    def stop(mod, args, out):
        raise _Stop

    hooks.append(layer.register_forward_hook(stop))
    try:
        for b in batches:
            try:
                model(b, use_cache=False)
            except _Stop:
                pass
    finally:
        for h in hooks:
            h.remove()
    return {g: torch.cat(v) for g, v in caps.items()}


def awq_scales(xmean: torch.Tensor, alpha: float) -> torch.Tensor:
    """llm-awq: s = x_mean^alpha clamped >= 1e-4, divided by sqrt(max*min); rounded to FP16 (stored value)."""
    s = xmean.float().pow(alpha).clamp(min=1e-4)
    s = s / (s.max() * s.min()).sqrt()
    return s.half()


def effective_weight(W: torch.Tensor, s: torch.Tensor, bits: int) -> tuple[torch.Tensor, object]:
    """Q(W diag(s)) diag(1/s) in FP32 -> FP16, plus the QResult of the scaled matrix."""
    sf = s.float()[None, :]
    qr = quantize(W.float() * sf, bits)
    return (qr.w_hat.float() / sf).half(), qr


@torch.no_grad()
def group_mse(X: torch.Tensor, Ws: list[torch.Tensor], s: torch.Tensor, bits: int, chunk: int = 8192) -> float:
    """Mean over tokens x outputs of (X W^T - X W_eff^T)^2 for the concatenated projections (no bias)."""
    D = torch.cat([(effective_weight(W, s, bits)[0].float() - W.float()) for W in Ws]).to(X.dtype)
    sse = 0.0
    for i in range(0, X.shape[0], chunk):
        y = X[i:i + chunk] @ D.T
        sse += float((y.float() ** 2).sum(dtype=torch.float64))
    return sse / (X.shape[0] * D.shape[0])


@torch.no_grad()
def search(model, batches_fn, bits_list=(8, 4, 3), log=print) -> dict:
    """Grid search for all layers/groups/bit widths. batches_fn() yields calibration batches."""
    L = model.config.num_hidden_layers
    nb = len(bits_list)
    mse = np.zeros((L, len(GROUPS), nb, len(ALPHAS)))
    best = np.zeros((L, len(GROUPS), nb), dtype=np.int64)
    scales: dict[str, np.ndarray] = {}
    t_cap = t_search = 0.0
    for li in range(L):
        t = time.perf_counter()
        X = capture_block_inputs(model, li, batches_fn())
        _sync()
        t_cap += time.perf_counter() - t
        t = time.perf_counter()
        mods = _mods(model.model.layers[li])
        for gi, (g, names) in enumerate(GROUPS):
            Ws = [mods[n].weight for n in names]
            xmean = X[g].abs().float().mean(0)
            for ai, alpha in enumerate(ALPHAS):
                s = awq_scales(xmean, alpha)
                for bi, b in enumerate(bits_list):
                    mse[li, gi, bi, ai] = group_mse(X[g], Ws, s, b)
            for bi, b in enumerate(bits_list):
                a = int(np.argmin(mse[li, gi, bi]))  # first minimum -> smaller alpha on ties
                best[li, gi, bi] = a
                scales[f"L{li}.{g}.b{b}"] = awq_scales(xmean, ALPHAS[a]).cpu().numpy()
        del X
        _sync()
        t_search += time.perf_counter() - t
        log(f"  awq layer {li}: alpha idx {best[li].tolist()}")
    return {"bits": list(bits_list), "alphas": ALPHAS, "groups": [g for g, _ in GROUPS], "mse": mse,
            "best_alpha_index": best, "scales": scales, "capture_s": t_cap, "search_s": t_search}


def save_search(path, res: dict) -> None:
    arr = {f"scale.{k}": v for k, v in res["scales"].items()}
    np.savez_compressed(path, bits=np.array(res["bits"]), alphas=np.array(res["alphas"]),
                        groups=np.array(res["groups"]), mse=res["mse"], best_alpha_index=res["best_alpha_index"],
                        capture_s=res["capture_s"], search_s=res["search_s"], **arr)


def load_search(path) -> dict:
    z = np.load(path)
    return {"bits": z["bits"].tolist(), "alphas": z["alphas"].tolist(), "groups": z["groups"].tolist(),
            "mse": z["mse"], "best_alpha_index": z["best_alpha_index"], "capture_s": float(z["capture_s"]),
            "search_s": float(z["search_s"]),
            "scales": {k[len("scale."):]: z[k] for k in z.files if k.startswith("scale.")}}


def layer_scales(res: dict, li: int, bits: int, device) -> dict[str, torch.Tensor]:
    return {g: torch.from_numpy(res["scales"][f"L{li}.{g}.b{bits}"]).to(device) for g, _ in GROUPS}


@torch.no_grad()
def apply_quantized(model, res, bits, on_matrix=None, layers=None, res_by_bits=None) -> dict:
    """Replace eligible weights by AWQ-style effective FP16 weights.

    bits: int (uniform) or a per-layer list. layers: subset of block indices (default all). res_by_bits: optional
    {bits: search-result} overriding `res` per bit width (Q3 uses saved Q2 scales for 3/4/8 and new ones for 2).
    on_matrix(qualname, QResult) is called for every quantized (scaled) matrix (packing check / byte counting).
    """
    L = len(model.model.layers)
    bl = [bits] * L if isinstance(bits, int) else list(bits)
    if len(bl) != L:
        raise ValueError("need one bit width per layer")
    todo = range(L) if layers is None else layers
    serialized_bytes = 0
    for li in todo:
        layer = model.model.layers[li]
        b = int(bl[li])
        r = res_by_bits[b] if res_by_bits else res
        mods = _mods(layer)
        S = layer_scales(r, li, b, mods["q_proj"].weight.device)
        # 1) fold down scales into up rows BEFORE up is scaled/quantized
        sd = S["down"].float()
        Wd = mods["down_proj"].weight
        qd = quantize(Wd.float() * sd[None, :], b)
        up_folded = mods["up_proj"].weight.float() / sd[:, None]
        if on_matrix:
            on_matrix(f"model.layers.{li}.mlp.down_proj", qd)
        new = {"down_proj": qd.w_hat}
        # 2) serialized groups: W_eff = Q(W diag(s)) diag(1/s)
        for g, names in GROUPS:
            if g == "down":
                continue
            for n in names:
                W = up_folded if n == "up_proj" else mods[n].weight
                weff, qr = effective_weight(W, S[g], b)
                new[n] = weff
                if on_matrix:
                    parent = "self_attn" if n in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                    on_matrix(f"model.layers.{li}.{parent}.{n}", qr)
            serialized_bytes += S[g].numel() * 2
        for n, w in new.items():
            mods[n].weight.copy_(w)
    n_l = len(list(todo))
    return {"serialized_scale_bytes": serialized_bytes,
            "serialized_scales": {"groups": list(SERIALIZED), "dtype": "float16",
                                  "per_layer_bytes": serialized_bytes // max(n_l, 1)}}


@torch.no_grad()
def verify_rescaling(model, pristine, res: dict, bits: int, batch: torch.Tensor) -> dict:
    """Unquantized check: scaled weights + down->up folding + explicit runtime input division for serialized
    groups, vs baseline logits. Also the merged-weight variant used by the fake-quant evaluation (unquantized)."""
    ref = model(batch, use_cache=False).logits.float()
    out = {}
    for variant in ("runtime_division", "merged_weights"):
        hooks = []
        for li, layer in enumerate(model.model.layers):
            mods = _mods(layer)
            S = layer_scales(res, li, bits, mods["q_proj"].weight.device)
            sd = S["down"].float()
            mods["up_proj"].weight.copy_((mods["up_proj"].weight.float() / sd[:, None]).half())
            mods["down_proj"].weight.copy_((mods["down_proj"].weight.float() * sd[None, :]).half())
            for g, names in GROUPS:
                if g == "down":
                    continue
                s = S[g]
                for n in names:
                    Ws = (mods[n].weight.float() * s.float()[None, :]).half()
                    if variant == "runtime_division":
                        mods[n].weight.copy_(Ws)
                        hooks.append(mods[n].register_forward_pre_hook(
                            lambda mod, args, s=s: (args[0] / s.to(args[0].dtype),) + tuple(args[1:])))
                    else:
                        mods[n].weight.copy_((Ws.float() / s.float()[None, :]).half())
        got = model(batch, use_cache=False).logits.float()
        for h in hooks:
            h.remove()
        pristine.restore(verify=False)
        out[variant] = {"max_abs_logit_diff": float((got - ref).abs().max()),
                        "mean_abs_logit_diff": float((got - ref).abs().mean()),
                        "max_abs_ref_logit": float(ref.abs().max()), "finite": bool(torch.isfinite(got).all())}
    pristine.verify()
    return out


def unfolded_equivalent(res: dict, bits, res_by_bits=None):
    """For output-error measurement: map a FOLDED Q2 weight back to the operator it replaces.
    down_proj stores Q(W_d diag(s_d)) and up_proj stores diag(1/s_d)*W_up_eff, so their isolated equivalents are
    Q(W_d diag(s_d)) diag(1/s_d) and diag(s_d) W_up_eff. Other projections are returned unchanged."""
    def eq(qualname: str, W: torch.Tensor) -> torch.Tensor:
        li = int(qualname.split(".")[2])
        if qualname.endswith("down_proj") or qualname.endswith("up_proj"):
            b = bits if isinstance(bits, int) else int(bits[li])
            r = res_by_bits[b] if res_by_bits else res
            sd = torch.from_numpy(r["scales"][f"L{li}.down.b{b}"]).to(W.device).float()
            if qualname.endswith("down_proj"):
                return W.float() / sd[None, :]
            return W.float() * sd[:, None]
        return W
    return eq
