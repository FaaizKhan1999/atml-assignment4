#!/usr/bin/env python3
"""Q3-proj (pre-registered second Q3 design, docs/decisions.md): projection-level allocation.

  probe : re-derive AWQ scales for --derive-bits and ASSERT equality with the saved npz files, then isolated dev
          probes dNLL(u, b) for every projection of blocks --blocks A B (inclusive), b in {2,3,4,8}.
          Writes results/task1/q3proj_probes_<tag>.json.
  solve : waits for the probe shards, exact MILP at B4/B3, dev check (predicted vs realised), re-measures the
          block-level allocation on dev, chooses the primary Q3 per budget (lower realised dev NLL; ties ->
          block-level). Writes results/task1/q3proj_allocation.json.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atml4.common.io import load_result, save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.seed import set_seed  # noqa: E402
from atml4.task1 import awq, data, p3, q3  # noqa: E402
from atml4.task1.calib import calibration_batches  # noqa: E402
from atml4.task1.model import Pristine, eligible_projections, load_base_model  # noqa: E402
from atml4.task1.quant import packed_nbytes  # noqa: E402

R = ROOT / "results/task1"
BUDGETS = {"B4": 1_147_982_848, "B3": 982_928_896}
BITS = q3.BITS
NAMES = awq.PROJ_NAMES


def saved_res():
    q2 = awq.load_search(R / "q2_awq_search.npz")
    q3s = awq.load_search(R / "q3_awq_search.npz")
    return {8: q2, 4: q2, 3: q2, 2: q3s}, q2, q3s


def meta(extra=None):
    return collect_meta(seed=6304, dtype=torch.float16, models=["base"],
                        extra={"dataset_selection": data.selection_record(["development", "calibration"]),
                               **(extra or {})})


def probe(a, model, pr, dev, recs):
    t_all = time.perf_counter()
    res_by_bits, q2, q3s = saved_res()
    L = model.config.num_hidden_layers
    # 1) re-derivation of the AWQ alphas/scales for the requested bit widths + assertion
    calib = data.load("calibration")
    res = awq.search(model, lambda: calibration_batches(calib, 8, dev), bits_list=tuple(a.derive_bits))
    pr.verify()
    match = {}
    for b in a.derive_bits:
        ref = q3s if b == 2 else q2
        i_new, i_old = res["bits"].index(b), ref["bits"].index(b)
        same_alpha = bool(np.array_equal(res["best_alpha_index"][:, :, i_new], ref["best_alpha_index"][:, :, i_old]))
        same_scales = all(np.array_equal(res["scales"][f"L{l}.{g}.b{b}"], ref["scales"][f"L{l}.{g}.b{b}"])
                          for l in range(L) for g in res["groups"])
        match[str(b)] = {"alpha_index_equal": same_alpha, "scales_equal": same_scales,
                         "reference": "q3_awq_search.npz" if b == 2 else "q2_awq_search.npz"}
    print("re-derivation match:", match, flush=True)
    assert all(m["alpha_index_equal"] and m["scales_equal"] for m in match.values()), match
    # 2) probes
    fp16 = p3.dev_nll(model, recs, dev)
    blocks = list(range(a.blocks[0], a.blocks[1] + 1))
    t = time.perf_counter()
    rows = []
    for l in blocks:
        for n in NAMES:
            d = {}
            for b in BITS:
                awq.apply_quantized_proj(model, {(l, n): b}, res_by_bits)
                d[str(b)] = p3.dev_nll(model, recs, dev)["mean_nll"] - fp16["mean_nll"]
                pr.restore_layers([l])
            rows.append({"layer": l, "proj": n, "dnll": d})
            print(f"probe L{l} {n:9s} " + " ".join(f"b{b}={d[str(b)]:+.5f}" for b in BITS), flush=True)
    torch.cuda.synchronize(dev)
    t_probe = time.perf_counter() - t
    pr.verify()
    save_result(R / f"q3proj_probes_{a.tag}.json",
                {"blocks": blocks, "bits_options": list(BITS), "dev_fp16": fp16, "probes": rows,
                 "rederivation": {"bits": list(a.derive_bits), "match": match, "capture_s": res["capture_s"],
                                  "search_s": res["search_s"]},
                 "gpu_seconds": {"total": time.perf_counter() - t_all, "probes": t_probe, "n_probes": 4 * len(rows)}},
                meta({"device": a.device, "tag": a.tag}))


def solve(a, model, pr, dev, recs):
    t_all = time.perf_counter()
    res_by_bits, _, _ = saved_res()
    L = model.config.num_hidden_layers
    files = [R / f"q3proj_probes_{t}.json" for t in a.shards]
    while not all(f.exists() for f in files):
        time.sleep(15)
    shards = [load_result(f)[1] for f in files]
    probes = {(r["layer"], r["proj"]): {int(b): v for b, v in r["dnll"].items()} for s in shards for r in s["probes"]}
    units = [(l, n) for l in range(L) for n in NAMES]
    assert set(probes) == set(units), "probe shards incomplete"
    shapes = {(p.layer, p.name): tuple(p.shape) for p in eligible_projections(model)}
    from atml4.task1 import storage

    st, _ = storage.model_storage_from_torch(model)
    costs = [probes[u] for u in units]
    weights = [{b: packed_nbytes(*shapes[u], b) for b in BITS} for u in units]
    groups = []
    for l in range(L):
        for g in awq.SERIALIZED:
            members = [i for i, u in enumerate(units) if u[0] == l and awq.PROJ_GROUP[u[1]] == g]
            vecs, vob = [], {}
            for b in BITS:
                v = res_by_bits[b]["scales"][f"L{l}.{g}.b{b}"]
                j = next((k for k, w in enumerate(vecs) if np.array_equal(v, w)), None)
                if j is None:
                    vecs.append(v)
                    j = len(vecs) - 1
                vob[b] = j
            groups.append({"units": members, "vec_of_bits": vob, "vec_bytes": {j: v.size * 2 for j, v in enumerate(vecs)},
                           "layer": l, "group": g})
    # sanity: uniform 4/3-bit reproduce the budgets exactly
    for name, b in (("B4", 4), ("B3", 3)):
        bm = {u: b for u in units}
        tot = st.other_bytes + sum(w[b] for w in weights) + awq.serialized_bytes_proj(bm, res_by_bits, range(L))
        assert tot == BUDGETS[name], (name, tot)
    fp16 = p3.dev_nll(model, recs, dev)
    blk = load_result(R / "q3_allocation.json")[1]["allocations"]
    allocs = {}
    for name, B in BUDGETS.items():
        t = time.perf_counter()
        sol = q3.proj_milp(costs, weights, [{k: v for k, v in g.items() if k in ("units", "vec_of_bits", "vec_bytes")}
                                            for g in groups], B, fixed_bytes=st.other_bytes)
        t_milp = time.perf_counter() - t
        bm = {u: b for u, b in zip(units, sol["bits"])}
        pr.restore(verify=False)
        info = awq.apply_quantized_proj(model, bm, res_by_bits)
        real = p3.dev_nll(model, recs, dev)
        pr.restore(verify=False)
        awq.apply_quantized(model, None, blk[name]["bits"], res_by_bits=res_by_bits)
        real_blk = p3.dev_nll(model, recs, dev)
        pr.restore(verify=False)
        packed = sum(weights[i][b] for i, b in enumerate(sol["bits"]))
        total = packed + info["serialized_scale_bytes"] + st.other_bytes
        assert total == sol["bytes"] and total <= B
        primary = "projection" if real["mean_nll"] < real_blk["mean_nll"] else "block"
        counts = {n: {str(b): sum(1 for (l, nn), bb in bm.items() if nn == n and bb == b) for b in BITS} for n in NAMES}
        per_block = [{"layer": l, "bits": {n: bm[(l, n)] for n in NAMES},
                      "packed_bytes": sum(weights[units.index((l, n))][bm[(l, n)]] for n in NAMES),
                      "param_weighted_mean_bits": sum(bm[(l, n)] * shapes[(l, n)][0] * shapes[(l, n)][1] for n in NAMES)
                      / sum(shapes[(l, n)][0] * shapes[(l, n)][1] for n in NAMES)} for l in range(L)]
        allocs[name] = {"budget_bytes": B, "bits_map": [{"layer": l, "proj": n, "bits": bm[(l, n)]} for l, n in units],
                        "counts_proj_type_x_bits": counts, "per_block": per_block,
                        "eligible_packed_bytes": packed, "serialized_awq_scale_bytes": info["serialized_scale_bytes"],
                        "other_params_bytes": st.other_bytes, "total_bytes": total, "slack_bytes": B - total,
                        "predicted_dev_dnll": sol["cost"], "realised_dev_dnll": real["mean_nll"] - fp16["mean_nll"],
                        "realised_dev_nll": real["mean_nll"], "realised_dev_ppl": real["ppl"],
                        "block_level_remeasured": {"bits": blk[name]["bits"], "dev_nll": real_blk["mean_nll"],
                                                   "dev_dnll": real_blk["mean_nll"] - fp16["mean_nll"],
                                                   "dev_ppl": real_blk["ppl"],
                                                   "saved_realised_dev_dnll": blk[name]["realised_dev_dnll"]},
                        "primary": primary,
                        "primary_rule": "lower realised dev NLL among {block, projection}; ties -> block",
                        "milp_seconds": t_milp, "milp_tie_break_status": sol["tie_break_status"]}
        print(name, {k: v for k, v in allocs[name].items() if k not in ("bits_map", "per_block")}, flush=True)
    pr.restore(verify=True)
    sens = [[probes[u][b] for b in BITS] for u in units]
    payload = {"units": [f"L{l}.{n}" for l, n in units], "bits_options": list(BITS), "sensitivity_dnll": sens,
               "dev_fp16_solve": fp16, "dev_fp16_probe_shards": {t: s["dev_fp16"] for t, s in zip(a.shards, shards)},
               "rederivation": {t: s["rederivation"] for t, s in zip(a.shards, shards)},
               "unit_bytes": [{str(b): w[b] for b in BITS} for w in weights],
               "allocations": allocs,
               "gpu_seconds": {"solve_total": time.perf_counter() - t_all,
                               "probe_shards": {t: s["gpu_seconds"] for t, s in zip(a.shards, shards)}},
               "assumption": "additive isolated per-projection probes (interactions incl. down->up fold ignored)"}
    save_result(R / "q3proj_allocation.json", payload, meta({"device": a.device}))
    np.savez_compressed(R / "q3proj_sensitivity.npz", dnll=np.array(sens), bits=np.array(BITS),
                        units=np.array(payload["units"]),
                        **{f"bits_{k}": np.array([x["bits"] for x in v["bits_map"]]) for k, v in allocs.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("probe", "solve"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--derive-bits", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--blocks", type=int, nargs=2, default=[0, 27])
    ap.add_argument("--tag", default="a")
    ap.add_argument("--shards", nargs="+", default=["a", "b"])
    a = ap.parse_args()
    set_seed(6304, deterministic=False)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    model, _ = load_base_model(a.device)
    pr = Pristine(model)
    recs = data.load("development")
    (probe if a.mode == "probe" else solve)(a, model, pr, dev, recs)


if __name__ == "__main__":
    main()
