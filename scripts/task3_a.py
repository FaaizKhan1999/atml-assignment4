#!/usr/bin/env python3
"""Task 3 kernel T3-a.

  checks     : real-model sampler checks (split prefill + decode == full forward for target and draft; speculative
               runs at gamma 2/4 with the per-round cache assertions; counters)            -> results/task3/a_checks.json
  bank3      : speculative prefix bank (iii) = our exact sampler, original student, gamma 4, max 192 new tokens,
               EOS-aware, course per-sequence seeds (teacher offset) and course position logic; bank name
               'exact_spec_original_gamma4' (course naming)                               -> results/task3/bank_spec.jsonl.gz
  collection : frozen off-policy teacher-response collection (800 training prompts)        -> results/task3/teacher_collection.jsonl.gz
  baseline   : EXCLUSIVE Step 2 timing (other GPU idle): warm setup, isolated draft/target single-step and gamma-token
               verification costs at contexts 64/256/512, 120 prompts x gamma {2,4} x 2 reps speculative with an
               adjacent target-only run (order alternated), EOS-aware cap 64; fixed-64 (EOS ignored) diagnostic on the
               first 24 GSM8K test IDs                                                    -> results/task3/baseline.jsonl, baseline_costs.json
Seeds: generation seed for (prompt j, rep r) = 6304 + 10007 j + 7919 (r + 1), shared by the speculative run and its
target-only reference (paired); bank (iii) uses the course seed 6304 + 10007 j.
"""
import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta, model_revision  # noqa: E402
from atml4.common.timing import read_peak_memory, reset_peak_memory  # noqa: E402
from atml4.task3 import data as D  # noqa: E402
from atml4.task3.speculative import HFLM, autoregressive_generate, speculative_generate  # noqa: E402

OUT = ROOT / "results/task3"


def load(name, device):
    mid, rev = model_revision(name)
    m = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.float16, attn_implementation="sdpa")
    return m.to(device).eval().requires_grad_(False)


def tokenizer():
    mid, rev = model_revision("target")
    return AutoTokenizer.from_pretrained(mid, revision=rev)


def gen_seed(j, r):
    return 6304 + 10007 * j + 7919 * (r + 1)


def meta(extra=None):
    return collect_meta(seed=6304, dtype=torch.float16, models=["target", "draft"], extra=extra or {})


@torch.no_grad()
def checks(a, dev, tok):
    T, Dm = load("target", dev), load("draft", dev)
    tp = D.test_prompts()
    out = {"split_vs_full": [], "spec_runs": []}
    for name, m in (("target", T), ("draft", Dm)):
        for p in (tp[0], tp[60], tp[90]):
            ids = D.chat_ids(tok, p["text"])
            g = torch.Generator(device=dev).manual_seed(1)
            r = autoregressive_generate(HFLM(m, dev), ids, 24, g, eos_ids=())
            seq = ids + r["tokens"]
            full = m(torch.tensor([seq], device=dev)).logits[0, len(ids) - 1:-1].float()
            lm = HFLM(m, dev)
            split = torch.cat([lm.feed(ids)[-1:], lm.feed(r["tokens"][:-1])])
            out["split_vs_full"].append({"model": name, "id": p["id"],
                                         "max_abs_logit_diff": float((full - split).abs().max()),
                                         "argmax_agree": bool((full.argmax(-1) == split.argmax(-1)).all())})
    for gamma in (2, 4):
        for j in (0, 60, 90):
            ids = D.chat_ids(tok, tp[j]["text"])
            g = torch.Generator(device=dev).manual_seed(gen_seed(j, 0))
            tl, dl = HFLM(T, dev), HFLM(Dm, dev)
            r = speculative_generate(tl, dl, ids, 64, gamma, g, eos_ids=D.EOS_IDS, device=dev)
            seq = ids + r["tokens"]
            assert tl.cached == seq[:-1] and tl.cache.get_seq_length() == len(seq) - 1
            out["spec_runs"].append({"gamma": gamma, "id": tp[j]["id"], "n_tokens": len(r["tokens"]),
                                     "stop": r["stop"], "stats": {k: v for k, v in r["stats"].items() if k != "timing"},
                                     "cache_len_target": tl.cache.get_seq_length(), "seq_len": len(seq),
                                     "cache_len_draft": dl.cache.get_seq_length(),
                                     "fully_accepted_rounds": sum(1 for k in r["stats"]["accepted_prefix"] if k == gamma),
                                     "text": tok.decode(r["tokens"])})
    out["all_split_checks_argmax_agree"] = all(x["argmax_agree"] for x in out["split_vs_full"])
    save_result(OUT / "a_checks.json", out, meta())
    print(json.dumps({k: v for k, v in out.items() if k != "spec_runs"}, indent=1)[:2000], flush=True)


@torch.no_grad()
def bank3(a, dev, tok):
    T, Dm = load("target", dev), load("draft", dev)
    recs, src = [], []
    t0 = time.time()
    for j, p in enumerate(D.test_prompts()):
        ids = D.chat_ids(tok, p["text"])
        seed = D.bank_seed(j, "teacher")
        g = torch.Generator(device=dev).manual_seed(seed)
        r = speculative_generate(HFLM(T, dev), HFLM(Dm, dev), ids, 192, 4, g, eos_ids=D.EOS_IDS, device=dev)
        recs += D.bank_records(p["id"], p["domain"], "exact_spec_original_gamma4", p["text"], r["tokens"], seed,
                               {"producing_checkpoint": "Qwen2.5-0.5B-Instruct (original) draft, gamma 4, target 1.5B"})
        src.append({"prompt_id": p["id"], "len": len(r["tokens"]), "stop": r["stop"],
                    "stats": {k: v for k, v in r["stats"].items() if k not in ("timing", "accepted_prefix")}})
        if (j + 1) % 20 == 0:
            print(f"bank3 {j + 1}/120 {time.time() - t0:.0f}s", flush=True)
    save_result(OUT / "bank_spec.jsonl.gz", recs, meta({"sources": src, "seconds": time.time() - t0}))


@torch.no_grad()
def collection(a, dev, tok):
    from atml4.task3.distill import teacher_collection

    T = load("target", dev)
    t0 = time.time()
    coll = teacher_collection(T, tok, D.train_prompts(), dev)
    torch.cuda.synchronize(dev)
    sec = time.time() - t0
    rows = [{"source_id": k, **v} for k, v in coll.items()]
    save_result(OUT / "teacher_collection.jsonl.gz", rows,
                meta({"teacher_generation_seconds": sec, "n": len(rows),
                      "tokens": sum(len(r["tokens"]) for r in rows), "batch": 16}))
    print(f"collection {len(rows)} responses in {sec:.0f}s", flush=True)


def _timed(fn, dev, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(dev)
    xs = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize(dev)
        xs.append(s.elapsed_time(e))
    return {"median_ms": statistics.median(xs), "samples_ms": xs, "reps": reps, "warmup": warm}


@torch.no_grad()
def isolated_costs(T, Dm, tok, dev) -> dict:
    tp = D.test_prompts()
    base = []
    for p in tp[:20]:
        base += D.chat_ids(tok, p["text"])
    out = {}
    for L in (64, 256, 512):
        ctx = base[:L]
        rec = {}
        for name, m in (("target", T), ("draft", Dm)):
            lm = HFLM(m, dev)
            lm.feed(ctx)

            def step(n=1, lm=lm):
                lm.feed(ctx[:n])
                lm.crop(L)
            rec[f"{name}_1tok"] = _timed(lambda: step(1), dev)
            if name == "target":
                for g in (2, 4):
                    rec[f"target_{g + 1}tok_verify"] = _timed(lambda g=g: step(g + 1), dev)
        t1 = rec["target_1tok"]["median_ms"]
        rec["c"] = rec["draft_1tok"]["median_ms"] / t1
        rec["v_gamma"] = {str(g): rec[f"target_{g + 1}tok_verify"]["median_ms"] / t1 for g in (2, 4)}
        out[str(L)] = rec
    return out


@torch.no_grad()
def baseline(a, dev, tok):
    from atml4.task1.resources import gpu_state

    T, Dm = load("target", dev), load("draft", dev)
    tp = D.test_prompts()
    # warm setup: first-ever generation calls after load (CUDA/cuBLAS init), recorded separately
    ids0 = D.chat_ids(tok, tp[0]["text"])
    t = time.perf_counter()
    speculative_generate(HFLM(T, dev), HFLM(Dm, dev), ids0, 16, 4, torch.Generator(device=dev).manual_seed(0),
                         eos_ids=D.EOS_IDS, timed=True, device=dev)
    warm_spec = time.perf_counter() - t
    t = time.perf_counter()
    autoregressive_generate(HFLM(T, dev), ids0, 16, torch.Generator(device=dev).manual_seed(0), timed=True, device=dev)
    warm_ar = time.perf_counter() - t
    for _ in range(3):  # further warmup, not recorded
        speculative_generate(HFLM(T, dev), HFLM(Dm, dev), ids0, 32, 2, torch.Generator(device=dev).manual_seed(1),
                             eos_ids=D.EOS_IDS, timed=True, device=dev)
        autoregressive_generate(HFLM(T, dev), ids0, 32, torch.Generator(device=dev).manual_seed(1), timed=True, device=dev)
    costs = isolated_costs(T, Dm, tok, dev)
    print("isolated costs", {L: (round(c["c"], 3), {g: round(v, 3) for g, v in c["v_gamma"].items()})
                             for L, c in costs.items()}, flush=True)
    rows = []

    def run_pair(j, p, rep, gamma, cap, eos, tag):
        ids = D.chat_ids(tok, p["text"])
        seed = gen_seed(j, rep)
        res = {}
        order = ("spec", "ar") if (j + rep + gamma) % 2 == 0 else ("ar", "spec")
        for kind in order:
            g = torch.Generator(device=dev).manual_seed(seed)
            reset_peak_memory(dev)
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            if kind == "spec":
                r = speculative_generate(HFLM(T, dev), HFLM(Dm, dev), ids, cap, gamma, g, eos_ids=eos, timed=True,
                                         device=dev)
            else:
                r = autoregressive_generate(HFLM(T, dev), ids, cap, g, eos_ids=eos, timed=True, device=dev)
            torch.cuda.synchronize(dev)
            tot = time.perf_counter() - t0
            n = len(r["tokens"])
            res[kind] = {"n_tokens": n, "stop": r["stop"], "total_s": tot, "prefill_s": r["prefill_s"],
                         "decode_s": tot - r["prefill_s"], "tokens_per_s_decode": n / max(tot - r["prefill_s"], 1e-9),
                         "tokens_per_s_total": n / tot, **read_peak_memory(dev)}
            if kind == "spec":
                st = r["stats"]
                res[kind].update({k: st[k] for k in ("rounds", "proposed", "examined", "accepted", "target_passes",
                                                     "draft_passes", "bonus", "residual", "one_token_rounds",
                                                     "accepted_over_examined", "accepted_over_proposed",
                                                     "mean_accepted_prefix", "timing")})
                res[kind]["target_passes_per_token"] = st["target_passes"] / max(n, 1)
            else:
                res[kind]["target_passes"] = r["target_passes"]
        rows.append({"tag": tag, "id": p["id"], "workload": p["workload"], "rep": rep, "gamma": gamma, "seed": seed,
                     "order": list(order), "cap": cap, "eos_aware": bool(eos), **res,
                     "time_ratio_spec_over_ar": res["spec"]["total_s"] / res["ar"]["total_s"],
                     "decode_tps_ratio_spec_over_ar": res["spec"]["tokens_per_s_decode"] / res["ar"]["tokens_per_s_decode"]})

    t0 = time.time()
    for rep in (0, 1):
        for gamma in (2, 4):
            for j, p in enumerate(tp):
                run_pair(j, p, rep, gamma, D.CAPS_SPEED, D.EOS_IDS, "primary_eos_aware_64")
            print(f"primary rep {rep} gamma {gamma} done {time.time() - t0:.0f}s", flush=True)
    diag = [p for p in tp if p["domain"] == "math"][:24]  # rule: first 24 GSM8K test IDs in manifest order
    for rep in (0, 1):
        for gamma in (2, 4):
            for j, p in enumerate(diag):
                run_pair(j, p, rep, gamma, 64, (), "diagnostic_fixed64_ignore_eos")
    print(f"baseline done {time.time() - t0:.0f}s", flush=True)
    m = meta({"timing_exclusive": True, "other_gpu_state": gpu_state()})
    save_result(OUT / "baseline.jsonl", rows, m)
    save_result(OUT / "baseline_costs.json", {"warm_setup_s": {"speculative_first_call": warm_spec,
                                                               "target_only_first_call": warm_ar},
                                              "isolated": costs, "seconds": time.time() - t0}, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("checks", "bank3", "collection", "baseline"))
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    {"checks": checks, "bank3": bank3, "collection": collection, "baseline": baseline}[a.mode](a, dev, tok)


if __name__ == "__main__":
    main()
