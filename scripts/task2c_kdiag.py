#!/usr/bin/env python3
"""Task 2C channel-wise-K INT4 anomaly diagnostics (a-e). DIAGNOSTIC ONLY (pre-RoPE keys are never a method).

On 2 prompts (KV-2048-start-0-00, KV-4096-middle-2-00): FP16 prefill of the context with the working SDPA model,
post-RoPE K and V from the cache, pre-RoPE K from a k_proj forward hook (output incl. bias). For INT4 channel-wise
and token-wise K (and token-wise V), with the ORIGINAL quantizer (Task 1 zero-point form, z clamped to [0, 15]) and
the FIXED quantizer (FP16 offset = rounded-down min, FP16 scale rounded up, codes against stored metadata):
  a. synthetic per-channel-offset axis test
  b. relative error; codes vs stored metadata vs exact-FP32 metadata; max |K| per layer; per-channel-group range vs
     magnitude and the fraction of channel groups that do not contain 0
  c. error localisation per layer, head, group index, top channels
  d. residual / group-boundary checks
  e. pre-RoPE vs post-RoPE channel-wise error
Writes results/task2/partC/k_quant_diagnostics.json and (not committed) <out>/kv_<id>.pt with K/V tensors.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task2.kv import kvquant as KQ  # noqa: E402
from atml4.task2.kv import prompts as PR  # noqa: E402
from atml4.task2.kv.kvcache import prefill  # noqa: E402
from atml4.task2.kv.model import load  # noqa: E402

IDS = ("KV-2048-start-0-00", "KV-4096-middle-2-00")
G = 64


def old_quant_groups(Gx, bits):
    """ORIGINAL (buggy for groups not containing 0): Task 1 zero-point form with z clamped to [0, 2^b-1]."""
    x = Gx.float()
    qmax = 2 ** bits - 1
    xmin, xmax = x.amin(-1), x.amax(-1)
    s = ((xmax - xmin) / qmax).clamp(min=2.0 ** -24).half().float()
    z = torch.clamp(torch.round(-xmin / s), 0, qmax)
    q = torch.clamp(torch.round(x / s[..., None]) + z[..., None], 0, qmax)
    return (q - z[..., None]) * s[..., None]


def exact_meta_groups(Gx, bits):
    """Fixed form but with UNROUNDED FP32 offset/scale (to isolate FP16 metadata rounding)."""
    x = Gx.float()
    qmax = 2 ** bits - 1
    o, mx = x.amin(-1), x.amax(-1)
    s = ((mx - o) / qmax).clamp(min=2.0 ** -24)
    q = torch.clamp(torch.round((x - o[..., None]) / s[..., None]), 0, qmax)
    return o[..., None] + q * s[..., None]


def xhat(X, layout, bits, kind):
    Gx = KQ._groups(X, layout)
    if kind == "old":
        y = old_quant_groups(Gx, bits)
    elif kind == "fixed":
        y = KQ.quantize_groups(Gx, bits)[3]
    else:
        y = exact_meta_groups(Gx, bits)
    return KQ._ungroup(y, layout, X.shape).half().float()


def rel(a, b):
    return float((a - b).norm() / b.norm())


def analyse(Ks, label):
    """Ks: per layer [H, nq, d] fp16 (quantized prefix only)."""
    out = {}
    for kind in ("old", "fixed", "exact_fp32_meta"):
        for layout in ("channel", "token"):
            per_layer, per_head, per_group, per_chan = [], [], None, None
            num = den = 0.0
            for li, K in enumerate(Ks):
                Kf = K.float()
                E = xhat(K, layout, 4, kind) - Kf
                per_layer.append(float(E.norm() / Kf.norm()))
                per_head.append([float(E[h].norm() / Kf[h].norm()) for h in range(K.shape[0])])
                ge = E.reshape(K.shape[0], -1, G, K.shape[2]).pow(2).sum(dim=(0, 2, 3))
                gk = Kf.reshape(K.shape[0], -1, G, K.shape[2]).pow(2).sum(dim=(0, 2, 3))
                per_group = ge if per_group is None else per_group + ge
                per_chan = E.pow(2).sum(dim=(0, 1)) if per_chan is None else per_chan + E.pow(2).sum(dim=(0, 1))
                num += float(E.pow(2).sum())
                den += float(Kf.pow(2).sum())
                if li == 0:
                    gk_tot = gk
                else:
                    gk_tot = gk_tot + gk
            pg = (per_group / gk_tot).sqrt()
            out[f"{kind}/{layout}"] = {
                "rel_all_layers": (num / den) ** 0.5, "rel_mean_over_layers": sum(per_layer) / len(per_layer),
                "per_layer": per_layer, "per_head": per_head,
                "per_group_index_rel": {"first_group": float(pg[0]), "rest_mean": float(pg[1:].mean()),
                                        "max": float(pg.max()), "argmax": int(pg.argmax())},
                "share_of_sq_error_in_first_group": float(per_group[0] / per_group.sum()),
                "top10_channels_share_of_sq_error": float(per_chan.topk(10).values.sum() / per_chan.sum()),
                "top10_channels": per_chan.topk(10).indices.tolist()}
    # ranges vs magnitudes for channel groups
    stats = []
    for li, K in enumerate(Ks):
        Gx = KQ._groups(K, "channel").float()  # (H, n_groups, d, 64)
        mn, mx = Gx.amin(-1), Gx.amax(-1)
        mag = Gx.abs().amax(-1)
        stats.append({"layer": li, "max_abs": float(K.float().abs().max()),
                      "frac_channel_groups_not_containing_0": float(((mn > 0) | (mx < 0)).float().mean()),
                      "median_range": float((mx - mn).median()), "median_magnitude": float(mag.median()),
                      "median_range_over_magnitude": float(((mx - mn) / mag.clamp(min=1e-6)).median())})
    tok = KQ._groups(Ks[0], "token").float()
    return out, stats, float(((tok.amin(-1) > 0) | (tok.amax(-1) < 0)).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-dir", default="kv_dumps")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    model, tok = load(dev)
    cases = {c["id"]: c for c in PR.load_cases()}
    res = {"synthetic_axis_test": {}, "prompts": {}}
    g = torch.Generator().manual_seed(0)
    off = torch.empty(128).uniform_(100, 400, generator=g) * torch.where(torch.rand(128, generator=g) < .5, -1, 1)
    Ksyn = (off[None, None, :] + torch.randn(2, 256, 128, generator=g)).half()
    for kind in ("old", "fixed"):
        for layout in ("channel", "token"):
            res["synthetic_axis_test"][f"{kind}/{layout}"] = rel(xhat(Ksyn, layout, 4, kind), Ksyn.float())
    Path(a.save_dir).mkdir(parents=True, exist_ok=True)
    for pid in IDS:
        p = PR.build(cases[pid], tok)
        pre = {}
        hooks = [l.self_attn.k_proj.register_forward_hook(lambda m, i, o, li=li: pre.__setitem__(li, o.detach()))
                 for li, l in enumerate(model.model.layers)]
        try:
            Ks, Vs, _ = prefill(model, p.context_ids, dev)
        finally:
            for h in hooks:
                h.remove()
        H, d = Ks[0].shape[1], Ks[0].shape[3]
        L = p.L
        nq = (L // G) * G
        post = [k[0, :, :nq].contiguous() for k in Ks]
        prer = [pre[li][0].view(L, H, d).transpose(0, 1)[:, :nq].contiguous() for li in range(len(Ks))]
        vals = [v[0, :, :nq].contiguous() for v in Vs]
        torch.save({"K_post": [k.cpu() for k in Ks], "V": [v.cpu() for v in Vs],
                    "K_pre": [x.cpu() for x in prer], "L": L, "id": pid}, Path(a.save_dir) / f"kv_{pid}.pt")
        r = {"L": L, "nq": nq, "residual": L - nq,
             "d_checks": {"layout": list(Ks[0].shape), "layout_is_B_Hkv_L_dh": list(Ks[0].shape) == [1, H, L, d],
                          "quantized_prefix_tokens": nq, "group_starts": [0, 64, 128, "...", nq - 64],
                          "residual_positions": [nq, L - 1], "residual_excluded_from_groups_and_error": True}}
        r["post_rope_K"], r["post_rope_K_channel_group_stats"], r["frac_token_groups_not_containing_0_L0"] = analyse(post, "post")
        r["pre_rope_K"], r["pre_rope_K_channel_group_stats"], _ = analyse(prer, "pre")
        r["V_token_wise"] = {kind: sum(rel(xhat(v, "token", 4, kind), v.float()) for v in vals) / len(vals)
                             for kind in ("old", "fixed")}
        res["prompts"][pid] = r
        print(pid, {k: round(v["rel_mean_over_layers"], 4) for k, v in r["post_rope_K"].items()},
              "pre:", {k: round(v["rel_mean_over_layers"], 4) for k, v in r["pre_rope_K"].items()},
              "V:", r["V_token_wise"], flush=True)
        del Ks, Vs, pre, post, prer, vals
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    save_result(ROOT / "results/task2/partC/k_quant_diagnostics.json", res,
                collect_meta(seed=6304, dtype=torch.float16, models=["target"], extra={"diagnostic_only": True}))


if __name__ == "__main__":
    main()
