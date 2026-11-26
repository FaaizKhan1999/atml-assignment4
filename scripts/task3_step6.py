#!/usr/bin/env python3
"""Task 3 Step 6 (approved 2026-11-26): speculative execution after distillation, ONE gamma per kernel, one GPU with
the other idle.

Phases (all on --device):
  merge_check (--merge-check): mean / max KL and argmax agreement of merged vs unmerged (peft, FP32 adapters under
      FP16 autocast) logits for the 4 students on the fixed batch used in distillation (first 4 GSM8K-train + first 4
      short-train prompts), every position. Untimed.
  isolated costs: c and v_gamma per draft at contexts 64/256/512 (20 reps after 3 warmups).
  primary: 5 drafts x 120 prompts x 2 reps, EOS-aware 64-token cap; draft order fixed on rep 0 and reversed on rep 1;
      every speculative run has an ADJACENT target-only run with the same seed (pair order alternated by parity);
      per-round synced decomposition (draft / verify / accept+residual / cache) from speculative_generate(timed=True);
      prefill and host overhead separately; SM clock, temperature and power sampled after every prompt.
  diagnostic: the first 24 GSM8K test IDs, fixed 64 tokens with EOS ignored, 5 drafts x 2 reps (timing only).
  unmerged (--unmerged RUN): that adapter through the peft wrapper on the 24 diagnostic prompts (wrapper overhead).
Merged checkpoints: merged/<run> in the repo or /kaggle/input/*/merged/<run> (output of atml4-t3-b2 mounted as a
kernel source). Writes results/task3/step6_g<gamma>.jsonl, step6_g<gamma>_diag.jsonl, step6_g<gamma>_costs.json,
merge_check.json.
"""
import argparse
import glob
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

OUT = ROOT / "results/task3"
RUNS = ["off_s6304", "off_s6305", "on_s6304", "on_s6305"]
DRAFTS = ["original"] + RUNS


def merged_dir(run):
    p = ROOT / "merged" / run
    if p.exists():
        return p
    hits = sorted(glob.glob(f"/kaggle/input/*/merged/{run}")) + sorted(glob.glob(f"/kaggle/input/**/merged/{run}",
                                                                                   recursive=True))
    if not hits:
        raise FileNotFoundError(f"merged checkpoint {run} not found (mount focusedkognition/atml4-t3-b2)")
    return Path(hits[0])


def load_draft(name, dev):
    from transformers import AutoModelForCausalLM

    if name == "original":
        return load("draft", dev)
    m = AutoModelForCausalLM.from_pretrained(merged_dir(name), dtype=torch.float16, attn_implementation="sdpa")
    return m.to(dev).eval().requires_grad_(False)


def load_unmerged(run, dev):
    from peft import PeftModel

    return PeftModel.from_pretrained(load("draft", dev), OUT / "adapters" / run).to(dev).eval()


@torch.no_grad()
def merge_check(dev, tok):
    tp = D.train_prompts()
    batch = [D.chat_ids(tok, p["text"]) for p in tp["math"][:4] + tp["short"][:4]]
    out = {}
    for run in RUNS:
        um, mg = load_unmerged(run, dev), load_draft(run, dev)
        kls, agree, n = [], 0, 0
        for ids in batch:
            x = torch.tensor([ids], device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                u = um(input_ids=x).logits[0].float()
            m = mg(input_ids=x).logits[0].float()
            lu, lm = torch.log_softmax(u, -1), torch.log_softmax(m, -1)
            kls += (lu.exp() * (lu - lm)).sum(-1).tolist()
            agree += int((u.argmax(-1) == m.argmax(-1)).sum())
            n += u.shape[0]
        kls_t = torch.tensor(kls)
        out[run] = {"mean_kl": float(kls_t.mean()), "median_kl": float(kls_t.median()), "max_kl": float(kls_t.max()),
                    "p99_kl": float(kls_t.quantile(0.99)), "argmax_agree": agree / n, "positions": n}
        print("merge_check", run, out[run], flush=True)
        del um, mg
        torch.cuda.empty_cache()
    save_result(OUT / "merge_check.json", out, meta({"batch": "first 4 GSM8K-train + first 4 short-train prompts"}))


@torch.no_grad()
def one(T, Dm, ids, gamma, seed, dev, kind, cap, eos):
    g = torch.Generator(device=dev).manual_seed(seed)
    reset_peak_memory(dev)
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    if kind == "spec":
        r = speculative_generate(HFLM(T, dev), HFLM(Dm, dev), ids, cap, gamma, g, eos_ids=eos, timed=True, device=dev)
    else:
        r = autoregressive_generate(HFLM(T, dev), ids, cap, g, eos_ids=eos, timed=True, device=dev)
    torch.cuda.synchronize(dev)
    tot = time.perf_counter() - t0
    rec = {"n_tokens": len(r["tokens"]), "stop": r["stop"], "total_s": tot, "prefill_s": r["prefill_s"],
           "decode_s": tot - r["prefill_s"], **read_peak_memory(dev)}
    if kind == "spec":
        st = r["stats"]
        rec.update({k: st[k] for k in ("rounds", "proposed", "examined", "accepted", "target_passes", "draft_passes",
                                       "bonus", "residual", "one_token_rounds", "accepted_over_examined",
                                       "accepted_over_proposed", "mean_accepted_prefix", "timing")})
        rec["host_overhead_s"] = tot - r["prefill_s"] - sum(st["timing"].values())
    return rec


def run_block(T, drafts, prompts, gamma, cap, eos, reps, dev, sampler, tag):
    rows = []
    names = list(drafts)
    t0 = time.time()
    for rep in range(reps):
        order = names if rep == 0 else names[::-1]
        for j, p in prompts:
            ids = D.chat_ids(TOK, p["text"])
            seed = gen_seed(j, rep)
            for ci, d in enumerate(order):
                pair = ("spec", "ar") if (j + ci + rep) % 2 == 0 else ("ar", "spec")
                res = {k: one(T, drafts[d], ids, gamma, seed, dev, k, cap, eos) for k in pair}
                rows.append({"tag": tag, "draft": d, "gamma": gamma, "id": p["id"], "workload": p["workload"],
                             "rep": rep, "seed": seed, "pair_order": list(pair), "draft_position": ci, **res,
                             "latency_ratio_spec_over_ar": res["spec"]["total_s"] / res["ar"]["total_s"],
                             "decode_tps_ratio": (res["spec"]["n_tokens"] / res["spec"]["decode_s"])
                             / (res["ar"]["n_tokens"] / res["ar"]["decode_s"])})
            rows[-1]["gpu_after_prompt"] = sampler()
        print(f"[{tag}] rep {rep} done {time.time() - t0:.0f}s", flush=True)
    return rows


TOK = None


def main():
    global TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gamma", type=int, required=True)
    ap.add_argument("--merge-check", action="store_true")
    ap.add_argument("--unmerged", default=None)
    a = ap.parse_args()
    from atml4.task1.resources import gpu_state
    from task1_retime import make_sampler

    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    TOK = tokenizer()
    sampler, _ = make_sampler(dev.index or 0)
    if a.merge_check:
        merge_check(dev, TOK)
    T = load("target", dev)
    drafts = {d: load_draft(d, dev) for d in DRAFTS}
    tp = D.test_prompts()
    ids0 = D.chat_ids(TOK, tp[0]["text"])
    for Dm in drafts.values():  # warmup, not recorded
        for _ in range(2):
            one(T, Dm, ids0, a.gamma, 0, dev, "spec", 32, D.EOS_IDS)
            one(T, Dm, ids0, a.gamma, 0, dev, "ar", 32, D.EOS_IDS)
    t = time.time()
    costs = {d: isolated_costs(T, Dm, TOK, dev) for d, Dm in drafts.items()}
    m = meta({"timing_exclusive": True, "other_gpu_state": gpu_state(), "gamma": a.gamma, "drafts": DRAFTS})
    save_result(OUT / f"step6_g{a.gamma}_costs.json", {"isolated": costs, "seconds": time.time() - t}, m)
    allp = list(enumerate(tp))
    rows = run_block(T, drafts, allp, a.gamma, D.CAPS_SPEED, D.EOS_IDS, 2, dev, sampler, "primary_eos_aware_64")
    save_result(OUT / f"step6_g{a.gamma}.jsonl", rows, m)
    diag_p = [(j, p) for j, p in allp if p["domain"] == "math"][:24]  # first 24 GSM8K test IDs (manifest order)
    diag = run_block(T, drafts, diag_p, a.gamma, 64, (), 2, dev, sampler, "diagnostic_fixed64_ignore_eos")
    if a.unmerged:
        um = {f"unmerged_{a.unmerged}": load_unmerged(a.unmerged, dev)}
        diag += run_block(T, um, diag_p, a.gamma, 64, (), 2, dev, sampler, "unmerged_adapter_fixed64")
    save_result(OUT / f"step6_g{a.gamma}_diag.jsonl", diag, m)
    print("step6 done", flush=True)


if __name__ == "__main__":
    main()
