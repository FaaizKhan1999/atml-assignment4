#!/usr/bin/env python3
"""Task 3 Step 6 (PREPARED; run only after reviewer approval): speculative execution after distillation.

For the given drafts (original and/or merged students from merged/<run>) x gamma {2, 4} x 120 test prompts x 2 reps,
EOS-aware 64-token cap: each speculative run is immediately followed or preceded by a target-only run with the same
seed (adjacent reference); the order of the draft x gamma conditions is rotated per prompt (cyclic Latin square over
prompts, reversed on rep 1) to counterbalance drift. Per-round synced timing (draft proposal / target verification /
acceptance+residual sampling / cache rollback-advance) comes from speculative_generate(timed=True); prefill and host
overhead (total - prefill - sum of round components) are reported separately. Isolated c and v_gamma per draft at
contexts 64/256/512. Optionally one UNMERGED adapter (peft wrapper) is timed separately (--unmerged RUN).
Must run with the other GPU idle (one GPU per kernel). Writes results/task3/step6_<tag>.jsonl and step6_costs_<tag>.json.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.timing import read_peak_memory, reset_peak_memory  # noqa: E402
from atml4.task3 import data as D  # noqa: E402
from atml4.task3.speculative import HFLM, autoregressive_generate, speculative_generate  # noqa: E402
from task3_a import gen_seed, isolated_costs, load, meta, tokenizer  # noqa: E402
from task3_b import load_draft  # noqa: E402

OUT = ROOT / "results/task3"


def load_unmerged(run, dev):
    from peft import PeftModel

    base = load("draft", dev)
    return PeftModel.from_pretrained(base, OUT / "adapters" / run).to(dev).eval()


@torch.no_grad()
def one(T, Dm, ids, gamma, seed, dev, kind):
    g = torch.Generator(device=dev).manual_seed(seed)
    reset_peak_memory(dev)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    if kind == "spec":
        r = speculative_generate(HFLM(T, dev), HFLM(Dm, dev), ids, D.CAPS_SPEED, gamma, g, eos_ids=D.EOS_IDS,
                                 timed=True, device=dev)
    else:
        r = autoregressive_generate(HFLM(T, dev), ids, D.CAPS_SPEED, g, eos_ids=D.EOS_IDS, timed=True, device=dev)
    torch.cuda.synchronize(dev)
    tot = time.perf_counter() - t0
    rec = {"n_tokens": len(r["tokens"]), "stop": r["stop"], "total_s": tot, "prefill_s": r["prefill_s"],
           **read_peak_memory(dev)}
    if kind == "spec":
        st = r["stats"]
        comp = sum(st["timing"].values())
        rec.update({k: st[k] for k in ("rounds", "proposed", "examined", "accepted", "target_passes", "draft_passes",
                                       "bonus", "residual", "one_token_rounds", "accepted_over_examined",
                                       "accepted_over_proposed", "mean_accepted_prefix", "timing")})
        rec["host_overhead_s"] = tot - r["prefill_s"] - comp
    else:
        rec["decode_s"] = r["decode_s"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--drafts", nargs="+", required=True)
    ap.add_argument("--unmerged", default=None)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    from atml4.task1.resources import gpu_state

    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    tok = tokenizer()
    T = load("target", dev)
    drafts = {d: load_draft(d, dev) for d in a.drafts}
    if a.unmerged:
        drafts[f"unmerged_{a.unmerged}"] = load_unmerged(a.unmerged, dev)
    tp = D.test_prompts()
    ids0 = D.chat_ids(tok, tp[0]["text"])
    for Dm in drafts.values():  # warmup (not recorded)
        for _ in range(2):
            one(T, Dm, ids0, 4, 0, dev, "spec")
    costs = {d: isolated_costs(T, Dm, tok, dev) for d, Dm in drafts.items()}
    conds = [(d, g) for d in drafts for g in (2, 4)]
    rows = []
    t0 = time.time()
    for rep in (0, 1):
        for j, p in enumerate(tp):
            ids = D.chat_ids(tok, p["text"])
            k = j % len(conds)
            order = conds[k:] + conds[:k]
            if rep == 1:
                order = order[::-1]
            for ci, (d, g) in enumerate(order):
                seed = gen_seed(j, rep)
                pair = ("spec", "ar") if (j + ci + rep) % 2 == 0 else ("ar", "spec")
                res = {kind: one(T, drafts[d], ids, g, seed, dev, kind) for kind in pair}
                rows.append({"draft": d, "gamma": g, "id": p["id"], "workload": p["workload"], "rep": rep,
                             "seed": seed, "pair_order": list(pair), "condition_position": ci, **res,
                             "time_ratio_ar_over_spec": res["ar"]["total_s"] / res["spec"]["total_s"]})
        print(f"rep {rep} done {time.time() - t0:.0f}s", flush=True)
    m = meta({"timing_exclusive": True, "other_gpu_state": gpu_state(), "drafts": list(drafts)})
    save_result(OUT / f"step6_{a.tag}.jsonl", rows, m)
    save_result(OUT / f"step6_costs_{a.tag}.json", {"isolated": costs, "seconds": time.time() - t0}, m)


if __name__ == "__main__":
    main()
