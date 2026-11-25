#!/usr/bin/env python3
"""Q3: AWQ search incl. b=2 (re-deriving b in {8,4,3} and checking against Q2), 112 isolated dev probes,
exact knapsack at the Q2 INT4/INT3 budgets, and dev check of predicted vs realised dNLL (pre-registered Q3-design).
Writes results/task1/q3_awq_search.npz and results/task1/q3_allocation.json."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.seed import set_seed  # noqa: E402
from atml4.task1 import awq, data, p3, q3, storage  # noqa: E402
from atml4.task1.calib import calibration_batches  # noqa: E402
from atml4.task1.model import Pristine, eligible_projections, load_base_model  # noqa: E402

BUDGETS = {"B4": 1_147_982_848, "B3": 982_928_896}

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
set_seed(6304, deterministic=False)
dev = torch.device(a.device)
torch.cuda.set_device(dev)
t_all = time.perf_counter()
model, _ = load_base_model(a.device)
pr = Pristine(model)
L = model.config.num_hidden_layers
recs = data.load("development")
calib = data.load("calibration")
fp16 = p3.dev_nll(model, recs, dev)

# 1) AWQ search for b in {8,4,3,2}
res = awq.search(model, lambda: calibration_batches(calib, 8, dev), bits_list=(8, 4, 3, 2))
awq.save_search(ROOT / "results/task1/q3_awq_search.npz", res)
q2 = awq.load_search(ROOT / "results/task1/q2_awq_search.npz")
match = {}
for b in (8, 4, 3):
    i_new, i_old = res["bits"].index(b), q2["bits"].index(b)
    same_alpha = bool(np.array_equal(res["best_alpha_index"][:, :, i_new], q2["best_alpha_index"][:, :, i_old]))
    same_scales = all(np.array_equal(res["scales"][f"L{l}.{g}.b{b}"], q2["scales"][f"L{l}.{g}.b{b}"])
                      for l in range(L) for g in res["groups"])
    match[str(b)] = {"alpha_index_equal": same_alpha, "scales_equal": same_scales}
print("re-derived alpha match vs Q2:", match, flush=True)
res_by_bits = {8: q2, 4: q2, 3: q2, 2: res}  # Q3 uses SAVED Q2 scales for 3/4/8 (pre-registered)
pr.verify()

# 2) probes
t = time.perf_counter()
dnll = np.zeros((L, len(q3.BITS)))
for l in range(L):
    for j, b in enumerate(q3.BITS):
        awq.apply_quantized(model, None, [b] * L, layers=[l], res_by_bits=res_by_bits)
        dnll[l, j] = p3.dev_nll(model, recs, dev)["mean_nll"] - fp16["mean_nll"]
        pr.restore_layers([l])
    print(f"probe block {l}: " + " ".join(f"b{b}={dnll[l, j]:+.5f}" for j, b in enumerate(q3.BITS)), flush=True)
torch.cuda.synchronize(dev)
t_probe = time.perf_counter() - t
pr.verify()

# 3) budgets and exact knapsack
st, _ = storage.model_storage_from_torch(model)
shapes = [p.shape for p in eligible_projections(model) if p.layer == 0]
ser_per_layer = 3 * model.config.hidden_size * 2
serialized = ser_per_layer * L
block_bytes = {b: q3.block_packed_bytes(shapes, b) for b in q3.BITS}
allocs = {}
for name, B in BUDGETS.items():
    budget_elig = B - st.other_bytes - serialized
    sol = q3.knapsack([{b: float(dnll[l, j]) for j, b in enumerate(q3.BITS)} for l in range(L)],
                      [block_bytes] * L, budget_elig)
    total = sol["bytes"] + serialized + st.other_bytes
    assert total <= B
    # dev check of the chosen allocation (realised vs predicted)
    pr.restore(verify=False)
    awq.apply_quantized(model, None, sol["bits"], res_by_bits=res_by_bits)
    real = p3.dev_nll(model, recs, dev)
    pr.restore(verify=False)
    allocs[name] = {"budget_bytes": B, "eligible_budget_bytes": budget_elig, "bits": sol["bits"],
                    "eligible_packed_bytes": sol["bytes"], "total_bytes": total, "slack_bytes": B - total,
                    "predicted_dev_dnll": sol["cost"], "realised_dev_dnll": real["mean_nll"] - fp16["mean_nll"],
                    "realised_dev_ppl": real["ppl"], "bit_counts": {str(b): sol["bits"].count(b) for b in q3.BITS}}
    print(name, allocs[name], flush=True)
# Q2 uniform on dev (for the dev-vs-test carry-over check)
q2_dev = {}
for b in (4, 3):
    pr.restore(verify=False)
    awq.apply_quantized(model, None, [b] * L, res_by_bits=res_by_bits)
    r = p3.dev_nll(model, recs, dev)
    q2_dev[f"int{b}"] = {"dev_mean_nll": r["mean_nll"], "dev_ppl": r["ppl"], "dev_dnll": r["mean_nll"] - fp16["mean_nll"],
                         "predicted_sum_probe_dnll": float(dnll[:, q3.BITS.index(b)].sum())}
pr.restore(verify=True)
payload = {"bits_options": list(q3.BITS), "probe_dnll": dnll.tolist(), "dev_fp16": fp16,
           "block_packed_bytes": {str(b): v for b, v in block_bytes.items()},
           "serialized_awq_scale_bytes": serialized, "other_params_bytes": st.other_bytes,
           "allocations": allocs, "q2_uniform_dev": q2_dev, "alpha_rederivation_match": match,
           "b2_chosen_alpha": {g: [res["alphas"][int(i)] for i in res["best_alpha_index"][:, gi, res["bits"].index(2)]]
                               for gi, g in enumerate(res["groups"])},
           "gpu_seconds": {"total": time.perf_counter() - t_all, "awq_capture": res["capture_s"],
                           "awq_search_4bits": res["search_s"], "probes": t_probe},
           "assumption": "additive isolated per-block probes (cross-block interaction ignored)"}
save_result(ROOT / "results/task1/q3_allocation.json", payload,
            collect_meta(seed=6304, dtype=torch.float16, models=["base"],
                         extra={"dataset_selection": data.selection_record(["development", "calibration"])}))
