"""Task 1 condition registry. Each condition: apply(model, ctx) -> info dict, run on freshly restored weights.

All conditions are evaluated by atml4.task1.runner unchanged. ctx: tokenizer, device, pristine, verify_pack, root.
"""
from __future__ import annotations

import time
from typing import Callable

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

CONDITIONS: dict[str, Callable] = {}
HELLASWAG_N: dict[str, int] = {}
FAMILY: dict[str, str] = {}


def register(name: str, hellaswag_n: int = 512, family: str = ""):
    def deco(fn):
        if name in CONDITIONS:
            raise ValueError(f"duplicate condition {name}")
        CONDITIONS[name] = fn
        HELLASWAG_N[name] = hellaswag_n
        FAMILY[name] = family
        return fn
    return deco


@register("baseline", hellaswag_n=2000, family="baseline")
def baseline(model, ctx) -> dict:
    """Unchanged FP16 weights."""
    return {"description": "unchanged FP16 Qwen2.5-1.5B base"}


# ---------------------------------------------------------------- pruning
_XNORM = None


def _xnorms(ctx):
    global _XNORM
    if _XNORM is None:
        from .calib import load_stats
        from .prune import xnorms_from_stats

        t = time.perf_counter()
        stats, rows = load_stats(ctx["root"] / "results/task1/baseline/calib_input_stats.npz")
        _XNORM = (xnorms_from_stats(stats), time.perf_counter() - t, len(rows))
    return _XNORM


@torch.no_grad()
def _prune(model, ctx, score_kind: str, sparsity: float | None, two_four: bool = False) -> dict:
    from .model import eligible_projections
    from .prune import magnitude_score, row_mask, two_four_mask, wanda_score
    from .storage import two_four_matrix_bytes

    t0 = time.perf_counter()
    stat_s = 0.0
    if score_kind == "wanda":
        xn, stat_s, n_rows = _xnorms(ctx)
    masks = {}
    for p in eligible_projections(model):
        W = p.module.weight
        sc = magnitude_score(W) if score_kind == "magnitude" else wanda_score(W, xn[p.qualname])
        masks[p.qualname] = two_four_mask(sc) if two_four else row_mask(sc, sparsity)
    _sync()
    t_score = time.perf_counter() - t0
    t1 = time.perf_counter()
    pruned, numel, per, tf_bytes = 0, 0, [], 0
    for p in eligible_projections(model):
        m = masks[p.qualname]
        k = int(m.sum())
        if two_four:
            assert k == m.numel() // 2, "2:4 must zero exactly half"
            tf_bytes += two_four_matrix_bytes(m.numel())["total_bytes"]
        else:
            assert k == round(sparsity * p.shape[1]) * p.shape[0], "exact per-row count"
        p.module.weight.masked_fill_(m, 0)
        pruned += k
        numel += m.numel()
        per.append({"qualname": p.qualname, "mask_pruned": k})
    _sync()
    info = {"family": "prune", "score": score_kind, "target_sparsity": 0.5 if two_four else sparsity,
            "pattern": "2:4 (groups of 4 contiguous input columns)" if two_four else "unstructured per-row",
            "tie_break": "stable sort: lower column index pruned first",
            "mask_pruned": pruned, "mask_pruned_fraction": pruned / numel, "per_matrix_mask_pruned": per,
            "compress_seconds": {"stat_load_s": stat_s, "score_and_mask_s": t_score - stat_s, "search_s": 0.0,
                                 "apply_s": time.perf_counter() - t1},
            "calibration_stats_source": "results/task1/baseline/calib_input_stats.npz" if score_kind == "wanda"
            else None}
    if two_four:
        info["storage_two_four_estimate"] = {
            "eligible_bytes": tf_bytes, "kind": "representation ESTIMATE: FP16 kept values + 2-bit index per kept "
            "value, per-matrix byte padding; no sparse-tensor-core execution on T4"}
    return info


for _s in (0.25, 0.50, 0.75):
    _n = 2000 if _s == 0.50 else 512
    register(f"p1_mag_s{int(_s * 100)}", _n, "prune")(lambda m, c, s=_s: _prune(m, c, "magnitude", s))
    register(f"p2_wanda_s{int(_s * 100)}", _n, "prune")(lambda m, c, s=_s: _prune(m, c, "wanda", s))
register("p2_wanda_2of4", 512, "prune")(lambda m, c: _prune(m, c, "wanda", None, two_four=True))


# ---------------------------------------------------------------- quantization
def _pack_checker(ctx, bits=None):
    from .quant import pack, packed_nbytes, unpack

    acc = {"matrices": 0, "packed_bytes": 0, "formula_bytes": 0, "roundtrip_exact": True, "degenerate_groups": 0,
           "pad_used": False, "per_matrix": []}

    def on_matrix(q, qr):
        d_out = qr.codes.shape[0]
        f = packed_nbytes(d_out, qr.d_in, qr.bits)
        acc["formula_bytes"] += f
        acc["degenerate_groups"] += qr.degenerate_groups
        acc["pad_used"] |= qr.pad > 0
        acc["matrices"] += 1
        nbytes = f
        if ctx.get("verify_pack", True):
            blob = pack(qr)
            nbytes = len(blob)
            c, z, s, w = unpack(blob, d_out, qr.d_in, qr.bits, device=qr.codes.device)
            ok = torch.equal(w, qr.w_hat) and torch.equal(c, qr.codes) and torch.equal(z, qr.zero) and \
                torch.equal(s.view(torch.int16), qr.scale.view(torch.int16))
            acc["roundtrip_exact"] &= bool(ok)
        acc["packed_bytes"] += nbytes
        acc["per_matrix"].append({"qualname": q, "shape": [d_out, qr.d_in], "bits": qr.bits, "packed_bytes": nbytes})
    return acc, on_matrix


@torch.no_grad()
def _q1(model, ctx, bits: int) -> dict:
    from .model import eligible_projections
    from .quant import quantize

    acc, on_matrix = _pack_checker(ctx, bits)
    t0 = time.perf_counter()
    t_pack = 0.0
    for p in eligible_projections(model):
        qr = quantize(p.module.weight, bits)
        tp = time.perf_counter()
        on_matrix(p.qualname, qr)
        t_pack += time.perf_counter() - tp
        p.module.weight.copy_(qr.w_hat)
    _sync()
    total = time.perf_counter() - t0
    return {"family": "quant", "method": "Q1 uniform", "bits": bits, "group_size": 128,
            "execution": "fake-quantized FP16 (dequantized weights, dense FP16 kernels)",
            "packing": {k: v for k, v in acc.items()}, "serialized_scale_bytes": 0,
            "compress_seconds": {"stat_s": 0.0, "search_s": 0.0, "apply_s": total - t_pack,
                                 "pack_verify_s": t_pack}}


_AWQ = None


@torch.no_grad()
def _q2(model, ctx, bits: int) -> dict:
    from . import awq, data
    from .calib import calibration_batches
    from .model import eligible_projections

    global _AWQ
    path = ctx["root"] / "results/task1/q2_awq_search.npz"
    shared = {}
    if _AWQ is None:
        if path.exists():
            _AWQ = awq.load_search(path)
            shared["source"] = f"loaded {path.name}"
        else:
            recs = data.load("calibration")
            res = awq.search(model, lambda: calibration_batches(recs, 8, ctx["device"]), bits_list=(8, 4, 3))
            awq.save_search(path, res)
            _AWQ = res
            shared["source"] = "searched in this process"
    res = _AWQ
    if bits not in res["bits"]:
        raise ValueError(f"no search for {bits} bits")
    bi = res["bits"].index(bits)
    verify = None
    if ctx.get("verify_pack", True):
        recs = data.load("calibration")
        batch = torch.tensor([r["token_ids"] for r in recs[:4]], device=ctx["device"])
        verify = awq.verify_rescaling(model, ctx["pristine"], res, bits, batch)
    acc, on_matrix = _pack_checker(ctx, bits)
    t0 = time.perf_counter()
    info = awq.apply_quantized(model, res, bits, on_matrix=on_matrix)
    ctx["outerr_equivalent"] = awq.unfolded_equivalent(res, bits)
    _sync()
    t_apply = time.perf_counter() - t0
    assert acc["matrices"] == len(eligible_projections(model))
    alpha = res["best_alpha_index"][:, :, bi]
    return {"family": "quant", "method": "Q2 AWQ-style scaling + Q1 quantizer", "bits": bits, "group_size": 128,
            "execution": "fake-quantized FP16 (dequantized, inverse scales merged into FP16 weights)",
            "packing": acc, **info, "fold_validity": awq.check_fold_validity(model),
            "alpha_grid": res["alphas"],
            "chosen_alpha": {g: [res["alphas"][int(a)] for a in alpha[:, gi]] for gi, g in enumerate(res["groups"])},
            "search_mse_best": {g: [float(res["mse"][li, gi, bi, alpha[li, gi]]) for li in range(alpha.shape[0])]
                                for gi, g in enumerate(res["groups"])},
            "search_mse_alpha0_q1": {g: [float(res["mse"][li, gi, bi, 0]) for li in range(alpha.shape[0])]
                                     for gi, g in enumerate(res["groups"])},
            "rescaling_verification_unquantized": verify, "search_shared": shared,
            "compress_seconds": {"stat_capture_s_shared_all_bits": res["capture_s"],
                                 "search_s_shared_all_bits": res["search_s"],
                                 "apply_and_pack_verify_s": t_apply}}


for _b in (8, 4, 3):
    _n = 512 if _b == 8 else 2000
    register(f"q1_int{_b}", _n, "quant")(lambda m, c, b=_b: _q1(m, c, b))
    register(f"q2_awq_int{_b}", _n, "quant")(lambda m, c, b=_b: _q2(m, c, b))

FIXED_CONDITIONS = ["p1_mag_s25", "p1_mag_s50", "p1_mag_s75", "p2_wanda_s25", "p2_wanda_s50", "p2_wanda_s75",
                    "p2_wanda_2of4", "q1_int8", "q1_int4", "q1_int3", "q2_awq_int8", "q2_awq_int4", "q2_awq_int3"]


# ---------------------------------------------------------------- P3 (OWL block allocation, pre-registered)
@torch.no_grad()
def _p3(model, ctx, S: float) -> dict:
    import json

    from . import p3

    t0 = time.perf_counter()
    sel = json.loads((ctx["root"] / "results/task1/p3_selection.json").read_text())["payload"]["selected"][str(S)]
    xn, stat_s, _ = _xnorms(ctx)
    D = p3.block_outlier_ratios(model, xn, Ms=(sel["M"],))[sel["M"]]
    t_score = time.perf_counter() - t0
    t1 = time.perf_counter()
    info = p3.apply_p3(model, xn, D, S, sel["lambda"])
    _sync()
    assert info["s_blocks"] == sel["s_blocks"], "allocation differs from the dev-selected one"
    return {"family": "prune", "score": "wanda", "method": "P3 OWL block allocation", "target_sparsity": S,
            "selected_M": sel["M"], "selected_lambda": sel["lambda"], **info,
            "compress_seconds": {"stat_load_s": stat_s, "sensitivity_s": t_score - stat_s, "search_s": 0.0,
                                 "apply_s": time.perf_counter() - t1,
                                 "selection_grid_s": "see results/task1/p3_selection.json gpu_seconds"}}


register("p3_owl_s50", 2000, "prune")(lambda m, c: _p3(m, c, 0.50))
register("p3_owl_s75", 512, "prune")(lambda m, c: _p3(m, c, 0.75))


# ---------------------------------------------------------------- Q3 (probe + exact knapsack, pre-registered)
_Q3RES = None


@torch.no_grad()
def _q3(model, ctx, budget: str) -> dict:
    import json

    from . import awq, storage

    global _Q3RES
    alloc = json.loads((ctx["root"] / "results/task1/q3_allocation.json").read_text())["payload"]
    a = alloc["allocations"][budget]
    if _Q3RES is None:
        q2 = awq.load_search(ctx["root"] / "results/task1/q2_awq_search.npz")
        new = awq.load_search(ctx["root"] / "results/task1/q3_awq_search.npz")
        _Q3RES = {8: q2, 4: q2, 3: q2, 2: new}
    acc, on_matrix = _pack_checker(ctx)
    t0 = time.perf_counter()
    info = awq.apply_quantized(model, None, a["bits"], on_matrix=on_matrix, res_by_bits=_Q3RES)
    _sync()
    t_apply = time.perf_counter() - t0
    ctx["outerr_equivalent"] = awq.unfolded_equivalent(None, a["bits"], res_by_bits=_Q3RES)
    st, _ = storage.model_storage_from_torch(model)
    total = acc["packed_bytes"] + info["serialized_scale_bytes"] + st.other_bytes
    assert total <= a["budget_bytes"], f"realised {total} > budget {a['budget_bytes']}"
    nominal = sum(b for b in a["bits"]) / len(a["bits"])
    return {"family": "quant", "method": f"Q3 mixed precision (probe + exact knapsack) @ {budget}",
            "bits": "mixed", "bits_per_block": a["bits"], "nominal_bits": nominal, "group_size": 128,
            "budget_bytes": a["budget_bytes"], "realised_total_bytes": total,
            "execution": "fake-quantized FP16 (dequantized, inverse scales merged into FP16 weights)",
            "packing": acc, **info, "predicted_dev_dnll": a["predicted_dev_dnll"],
            "realised_dev_dnll": a["realised_dev_dnll"],
            "compress_seconds": {"apply_and_pack_verify_s": t_apply,
                                 "probe_and_search_s": "see results/task1/q3_allocation.json gpu_seconds"}}


register("q3_mixed_b4", 2000, "quant")(lambda m, c: _q3(m, c, "B4"))
register("q3_mixed_b3", 2000, "quant")(lambda m, c: _q3(m, c, "B3"))
NEW_CONDITIONS = ["p3_owl_s50", "p3_owl_s75", "q3_mixed_b4", "q3_mixed_b3"]


# ---------------------------------------------------------------- Q3-proj (second pre-registered Q3 design)
@torch.no_grad()
def _q3proj(model, ctx, budget: str) -> dict:
    import json

    from . import awq, storage

    global _Q3RES
    alloc = json.loads((ctx["root"] / "results/task1/q3proj_allocation.json").read_text())["payload"]
    a = alloc["allocations"][budget]
    bm = {(x["layer"], x["proj"]): x["bits"] for x in a["bits_map"]}
    if _Q3RES is None:
        q2 = awq.load_search(ctx["root"] / "results/task1/q2_awq_search.npz")
        new = awq.load_search(ctx["root"] / "results/task1/q3_awq_search.npz")
        _Q3RES = {8: q2, 4: q2, 3: q2, 2: new}
    acc, on_matrix = _pack_checker(ctx)
    t0 = time.perf_counter()
    info = awq.apply_quantized_proj(model, bm, _Q3RES, on_matrix=on_matrix)
    _sync()
    t_apply = time.perf_counter() - t0
    ctx["outerr_equivalent"] = awq.unfolded_equivalent_proj(bm, _Q3RES)
    st, per = storage.model_storage_from_torch(model)
    assert acc["matrices"] == len(per)
    total = acc["packed_bytes"] + info["serialized_scale_bytes"] + st.other_bytes
    assert total <= a["budget_bytes"], f"realised {total} > budget {a['budget_bytes']}"
    assert total == a["total_bytes"], "realised bytes differ from the allocation record"
    numel = {m["qualname"]: m["numel"] for m in per}
    nominal = sum(m["bits"] * numel[m["qualname"]] for m in acc["per_matrix"]) / sum(numel.values())
    return {"family": "quant", "method": f"Q3-proj mixed precision (per-projection probes + exact MILP) @ {budget}",
            "bits": "mixed", "counts_proj_type_x_bits": a["counts_proj_type_x_bits"],
            "nominal_bits": nominal, "group_size": 128, "budget_bytes": a["budget_bytes"],
            "realised_total_bytes": total,
            "execution": "fake-quantized FP16 (dequantized, inverse scales merged into FP16 weights)",
            "packing": acc, **info, "predicted_dev_dnll": a["predicted_dev_dnll"],
            "realised_dev_dnll": a["realised_dev_dnll"], "primary_at_budget": a["primary"],
            "compress_seconds": {"apply_and_pack_verify_s": t_apply,
                                 "probe_and_search_s": "see results/task1/q3proj_allocation.json gpu_seconds"}}


register("q3proj_b4", 2000, "quant")(lambda m, c: _q3proj(m, c, "B4"))
register("q3proj_b3", 2000, "quant")(lambda m, c: _q3proj(m, c, "B3"))
Q3PROJ_CONDITIONS = ["q3proj_b4", "q3proj_b3"]
