#!/usr/bin/env python3
"""Task 2 Part A (A1-A3): dense causal, tiled online-softmax and PyTorch SDPA on identical synthetic tensors.

B=2, H=4, d=64, L in {256,...,8192}, FP16 and FP32; tensors generated in FP32 on CUDA with seed 6304 + L and cast
(so FP16 and FP32 share values up to rounding). Reference: exact dense causal attention in FP32 computed in query
chunks (feasible at every L). Timing: 10 warmed reps (3 warmup), CUDA events, raw samples; forward-only and
forward+backward stored separately; OOM / unsupported recorded as rows. Must run with the other GPU idle.
Outputs: results/task2/partA/{correctness.json, timing.jsonl, backend_table.json}.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.timing import read_peak_memory, reset_peak_memory, time_cuda  # noqa: E402
from atml4.task1.resources import gpu_state  # noqa: E402
from atml4.task2 import attention as A  # noqa: E402

B, H, D = 2, 4, 64
LS = [256, 512, 1024, 2048, 4096, 8192]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
TILES = [64, 128, 256]
OUT = ROOT / "results/task2/partA"


def tensors(L, dtype, dev):
    g = torch.Generator(device=dev).manual_seed(6304 + L)
    q, k, v, go = (torch.randn(B, H, L, D, device=dev, generator=g) for _ in range(4))
    return [t.to(dtype) for t in (q, k, v, go)], 6304 + L


def ops():
    o = {"dense": A.dense_attention, "sdpa_default": A.sdpa}
    for t in TILES:
        o[f"tiled{t}"] = (lambda tt: (lambda q, k, v: A.tiled_attention(q, k, v, tt)))(t)
    return o


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def fwd_bwd(fn, q, k, v, go):
    qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out = fn(qq, kk, vv)
    out.backward(go)
    return out.detach(), (qq.grad, kk.grad, vv.grad)


def err(a, b):
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rel_fro": float(torch.linalg.vector_norm(a.float() - b.float()) / torch.linalg.vector_norm(b.float())),
            "finite": bool(torch.isfinite(a).all())}


def sdpa_backend(q, k, v):
    from torch.profiler import ProfilerActivity, profile

    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as p:
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
    names = [e.key for e in p.key_averages()]
    j = " ".join(names).lower()
    kind = "flash" if "flash" in j else "efficient" if ("fmha" in j or "efficient" in j) else \
        "cudnn" if "cudnn" in j else "math"
    try:
        choice = torch.nn.attention.SDPBackend(torch._fused_sdp_choice(q, k, v, is_causal=True)).name
    except Exception as e:  # pragma: no cover
        choice = f"unavailable: {e}"
    return {"profiled": kind, "fused_sdp_choice": choice,
            "kernels": [n for n in names if any(s in n.lower() for s in ("fmha", "flash", "attention", "softmax", "gemm"))][:8]}


def edge_cases(dev):
    """(i) extreme logits: stable (max-subtracted) vs naive softmax. Target max post-scale logit 1e4 in FP32; in
    FP16 the raw QK^T (= logit * sqrt(d) = 8e4) would overflow the FP16 range (65504), so FP16 uses 5e3."""
    out = {}
    g = torch.Generator(device=dev).manual_seed(6304)
    q0 = torch.randn(1, 1, 64, D, device=dev, generator=g)
    k0 = torch.randn(1, 1, 64, D, device=dev, generator=g)
    v0 = torch.randn(1, 1, 64, D, device=dev, generator=g)
    s0 = float((q0 @ k0.transpose(-1, -2)).abs().max()) / D ** 0.5
    for name, dt, target in (("fp32", torch.float32, 1e4), ("fp16", torch.float16, 5e3)):
        c = (target / s0) ** 0.5
        q, k, v = (q0 * c).to(dt), (k0 * c).to(dt), v0.to(dt)
        ref = A.dense_attention(q.double(), k.double(), v.double())
        rec = {"target_max_logit": target}
        for stable in (True, False):
            o = A.dense_attention(q, k, v, stable=stable)
            rec["stable" if stable else "naive"] = {
                "finite": bool(torch.isfinite(o).all()), "nan_count": int(torch.isnan(o).sum()),
                "max_abs_err_vs_fp64": float((o.double() - ref).abs().nan_to_num(float("inf")).max())}
        out[f"extreme_scores_{name}"] = rec
    # (ii) all-masked rows: causal AND key padding masking keys 0..3 -> rows 0..3 have no admissible key
    L = 8
    q, k, v = (torch.randn(1, 1, L, D, device=dev, generator=g).requires_grad_(True) for _ in range(3))  # noqa
    kpm = torch.ones(1, L, dtype=torch.bool, device=dev)
    kpm[:, :4] = False
    o = A.dense_attention(q, k, v, key_padding_mask=kpm)
    o.sum().backward()
    out["all_masked_rows"] = {"definition": "rows with no admissible key return 0 and get zero gradients",
                              "rows": [0, 1, 2, 3], "out_zero": bool((o[..., :4, :] == 0).all()),
                              "dq_zero_on_masked_rows": bool((q.grad[..., :4, :] == 0).all()),
                              "masked_keys_dk_dv_zero": bool((k.grad[..., :4, :] == 0).all() and
                                                             (v.grad[..., :4, :] == 0).all()),
                              "all_finite": bool(torch.isfinite(o).all() and torch.isfinite(q.grad).all())}
    return out


def causality(dev):
    res = {}
    for name, dt in DTYPES.items():
        (q, k, v, _), _ = tensors(512, dt, dev)
        t0 = 300
        k2, v2 = k.clone(), v.clone()
        k2[..., t0 + 1:, :] += 5.0
        v2[..., t0 + 1:, :] -= 7.0
        for op, fn in ops().items():
            with torch.no_grad():
                a, b = fn(q, k, v), fn(q, k2, v2)
            res[f"{op}_{name}"] = {"prefix_bit_identical": bool(torch.equal(a[..., :t0 + 1, :], b[..., :t0 + 1, :])),
                                   "suffix_changed": bool(not torch.equal(a[..., t0 + 1:, :], b[..., t0 + 1:, :]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    meta = collect_meta(seed=6304, extra={"timing_exclusive": True, "other_gpu_state": gpu_state(),
                                          "B": B, "H": H, "d": D, "Ls": LS, "tiles": TILES})
    t_all = time.time()
    correct = {"edge_cases": edge_cases(dev), "causality": causality(dev), "grid": []}
    backend = {"rows": [], "default_dispatch": []}
    timing = []
    for L in LS:
        for dname, dt in DTYPES.items():
            (q, k, v, go), seed = tensors(L, dt, dev)
            torch.cuda.empty_cache()
            ref_out, ref_g = A.dense_reference_fp32(q, k, v, go)
            for op, fn in ops().items():
                row = {"L": L, "dtype": dname, "op": op, "seed": seed}
                try:
                    out, gr = fwd_bwd(fn, q, k, v, go)
                    row.update({"status": "ok", "out": err(out, ref_out),
                                **{f"d{n}": err(g, r) for n, g, r in zip("qkv", gr, ref_g)}})
                    del out, gr
                except Exception as e:
                    row.update({"status": "OOM" if is_oom(e) else "error", "error": str(e)[:300]})
                torch.cuda.empty_cache()
                correct["grid"].append(row)
                for mode in ("forward", "forward_backward"):
                    trow = {"L": L, "dtype": dname, "op": op, "mode": mode, "unit": "ms", "reps": a.reps,
                            "warmup": 3, "seed": seed}
                    try:
                        if mode == "forward":
                            def f():
                                with torch.no_grad():
                                    fn(q, k, v)
                        else:
                            qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

                            def f():
                                qq.grad = kk.grad = vv.grad = None
                                fn(qq, kk, vv).backward(go)
                        reset_peak_memory(dev)
                        r = time_cuda(f, reps=a.reps, warmup=3, min_reps=10, device=dev)
                        trow.update({"status": "ok", **r.to_dict(), **read_peak_memory(dev)})
                    except Exception as e:
                        trow.update({"status": "OOM" if is_oom(e) else "error", "error": str(e)[:300]})
                    if op.startswith("tiled"):
                        trow["analytic"] = A.tiled_intermediate_bytes(B, H, L, D, int(op[5:]),
                                                                      2 if dname == "fp16" else 4)
                    if op == "dense":
                        trow["analytic"] = {"dense_score_matrix_bytes": B * H * L * L * (2 if dname == "fp16" else 4)}
                    timing.append(trow)
                    torch.cuda.empty_cache()
                    print(f"L={L} {dname} {op:12s} {mode:16s} {trow['status']} "
                          f"{trow.get('median', float('nan')):.3f} ms", flush=True)
            backend["default_dispatch"].append({"L": L, "dtype": dname, **sdpa_backend(q, k, v)})
            from torch.nn.attention import SDPBackend, sdpa_kernel
            for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                       SDPBackend.MATH):
                rec = {"L": L, "dtype": dname, "backend": be.name}
                for mode in ("forward", "forward_backward"):
                    try:
                        with sdpa_kernel(be):
                            if mode == "forward":
                                with torch.no_grad():
                                    o = A.sdpa(q, k, v)
                            else:
                                o, _ = fwd_bwd(A.sdpa, q, k, v, go)
                        rec[mode] = {"supported": True, "max_abs_err_vs_ref": float((o.float() - ref_out).abs().max())}
                    except Exception as e:
                        rec[mode] = {"supported": False, "status": "OOM" if is_oom(e) else "unsupported",
                                     "error": str(e)[:200]}
                    torch.cuda.empty_cache()
                backend["rows"].append(rec)
            del q, k, v, go, ref_out, ref_g
            torch.cuda.empty_cache()
    correct["seconds"] = time.time() - t_all
    save_result(OUT / "correctness.json", correct, meta)
    save_result(OUT / "backend_table.json", backend, meta)
    save_result(OUT / "timing.jsonl", timing, meta)
    print(f"Part A done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
