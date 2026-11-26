#!/usr/bin/env python3
"""Task 3 kernel T3-b (quality only; no timing claims).

  distill  --method off|on --seed S : forward-KL LoRA distillation from the ORIGINAL student; adapter ->
                                      results/task3/adapters/<run>/, log -> results/task3/distill_<run>.json; merge into
                                      a FRESH FP16 backbone -> merged/<run>/ (Kaggle output only) + merged-vs-unmerged
                                      logits check on a fixed prompt batch
  ownbank  --run R                  : frozen own-rollout prefix bank of merged student R on the 120 test prompts
                                      (temperature 1, max 192, EOS-aware, course seeds + position logic; bank name
                                      own_<R>; producing checkpoint recorded) -> results/task3/bank_own_<R>.jsonl.gz
  align                             : Step 5 metrics for the 5 drafts on every bank -> results/task3/align_rows.jsonl.gz,
                                      align_summary.json
  accuracy --models ...             : standalone accuracy (caps 512 math / 32 short, T = 1, 2 paired reps) ->
                                      results/task3/accuracy_<model>.jsonl
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from atml4.common.io import load_jsonl_result, save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.timing import read_peak_memory, reset_peak_memory  # noqa: E402
from atml4.task3 import data as D  # noqa: E402

OUT = ROOT / "results/task3"
MERGED = ROOT / "merged"
RUNS = ["off_s6304", "off_s6305", "on_s6304", "on_s6305"]
DRAFTS = ["original"] + RUNS


def _load(name, dev):
    sys.path.insert(0, str(ROOT / "scripts"))
    from task3_a import load
    return load(name, dev)


def _tok():
    sys.path.insert(0, str(ROOT / "scripts"))
    from task3_a import tokenizer
    return tokenizer()


def load_draft(name, dev):
    from transformers import AutoModelForCausalLM

    if name == "original":
        return _load("draft", dev)
    m = AutoModelForCausalLM.from_pretrained(MERGED / name, dtype=torch.float16, attn_implementation="sdpa")
    return m.to(dev).eval().requires_grad_(False)


def meta(extra=None):
    return collect_meta(seed=6304, dtype=torch.float16, models=["target", "draft"], extra=extra or {})


def distill(a, dev, tok):
    from atml4.task3 import distill as DS

    run = f"{a.method}_s{a.seed}"
    teacher = _load("target", dev)
    reset_peak_memory(dev)
    student = DS.add_lora(_load("draft", dev), a.seed)
    student.print_trainable_parameters()
    coll = None
    if a.method == "off":
        _, rows = load_jsonl_result(OUT / "teacher_collection.jsonl.gz")
        coll = {r["source_id"]: r for r in rows}
    t0 = time.time()
    tr = DS.Trainer(a.method, a.seed, teacher, student, tok, D.train_prompts(), dev, collection=coll,
                    log=lambda m: print(m, flush=True))
    res = tr.train()
    train_wall = time.time() - t0
    peak = read_peak_memory(dev)
    adir = OUT / "adapters" / run
    student.save_pretrained(adir)
    # merge into a FRESH FP16 backbone (timed) and check merged vs unmerged logits on a fixed prompt batch
    t = time.time()
    merged = DS.merge_adapter(_load("draft", dev), adir)
    torch.cuda.synchronize(dev)
    merge_s = time.time() - t
    MERGED.mkdir(exist_ok=True)
    merged.save_pretrained(MERGED / run)
    tok.save_pretrained(MERGED / run)
    tp = D.train_prompts()
    batch = [D.chat_ids(tok, p["text"]) for p in tp["math"][:4] + tp["short"][:4]]  # fixed: first 4 + 4 train prompts
    student.eval()
    errs = []
    with torch.no_grad():
        for ids in batch:
            x = torch.tensor([ids], device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                u = student(input_ids=x).logits[0].float()
            mg = merged(input_ids=x).logits[0].float()
            lu, lm = torch.log_softmax(u, -1), torch.log_softmax(mg, -1)
            errs.append({"max_abs_logit_diff": float((u - mg).abs().max()), "mean_abs_logit_diff": float((u - mg).abs().mean()),
                         "max_kl_unmerged_merged": float((lu.exp() * (lu - lm)).sum(-1).max()),
                         "argmax_agree_frac": float((u.argmax(-1) == mg.argmax(-1)).float().mean())})
    out = {"run": run, "method": a.method, "seed": a.seed, **res, "train_wall_s": train_wall,
           "params": DS.param_counts(student), "peak_memory": peak, "merge_s": merge_s,
           "merged_vs_unmerged": errs, "updates_log": tr.updates, "presentations": tr.presentations,
           "settings": {"lora": DS.LORA, "lr": 3e-4, "weight_decay": 0.01, "lr_schedule": "constant",
                        "tokens_per_update": DS.TOKENS_PER_UPDATE, "quota": DS.QUOTA, "clip": 1.0,
                        "amp": "fp16 autocast + dynamic GradScaler (unscale before clipping)"}}
    save_result(OUT / f"distill_{run}.json", out, meta({"run": run, "device": str(dev)}))
    print(f"{run}: {res} merge {merge_s:.1f}s merged-vs-unmerged max|dlogit| "
          f"{max(e['max_abs_logit_diff'] for e in errs):.4f}", flush=True)


@torch.no_grad()
def ownbank(a, dev, tok):
    from atml4.task3.generate import batched_sample

    m = load_draft(a.run, dev)
    tp = D.test_prompts()
    recs = []
    t0 = time.time()
    for b in range(0, 120, 16):
        chunk = list(range(b, min(b + 16, 120)))
        res = batched_sample(m, [D.chat_ids(tok, tp[j]["text"]) for j in chunk], [192] * len(chunk),
                             [D.bank_seed(j, "teacher") for j in chunk], D.EOS_IDS, dev, pad_id=tok.pad_token_id)
        for j, r in zip(chunk, res):
            recs += D.bank_records(tp[j]["id"], tp[j]["domain"], f"own_{a.run}", tp[j]["text"], r["tokens"],
                                   D.bank_seed(j, "teacher"), {"producing_checkpoint": f"merged/{a.run}"})
    save_result(OUT / f"bank_own_{a.run}.jsonl.gz", recs, meta({"seconds": time.time() - t0, "run": a.run}))


@torch.no_grad()
def align(a, dev, tok):
    from atml4.task3 import align as AL

    target = _load("target", dev)
    drafts = {d: load_draft(d, dev) for d in DRAFTS}
    banks = D.load_course_banks()
    _, spec = load_jsonl_result(OUT / "bank_spec.jsonl.gz")
    banks += spec
    for r in RUNS:
        banks += load_jsonl_result(OUT / f"bank_own_{r}.jsonl.gz")[1]
    groups = AL.group_bank(banks)
    wl = {p["id"]: p["workload"] for p in D.test_prompts()}
    t0 = time.time()
    rows = AL.evaluate(target, drafts, groups, lambda text: D.chat_ids(tok, text), wl, dev)
    save_result(OUT / "align_rows.jsonl.gz", rows, meta({"seconds": time.time() - t0}))
    print(f"align rows {len(rows)} in {time.time() - t0:.0f}s", flush=True)


@torch.no_grad()
def accuracy(a, dev, tok):
    from atml4.task3.generate import batched_sample

    tp = D.test_prompts()
    for name in a.models:
        m = load_draft(name, dev)
        rows = []
        t0 = time.time()
        for rep in (0, 1):
            for dom, bs in (("math", 16), ("short", 30)):
                idx = [j for j, p in enumerate(tp) if p["domain"] == dom]
                for b in range(0, len(idx), bs):
                    chunk = idx[b:b + bs]
                    seeds = [6304 + 10007 * j + 7919 * (rep + 1) for j in chunk]
                    res = batched_sample(m, [D.chat_ids(tok, tp[j]["text"]) for j in chunk],
                                         [D.CAPS_QUALITY[dom]] * len(chunk), seeds, D.EOS_IDS, dev,
                                         pad_id=tok.pad_token_id)
                    for j, s, r in zip(chunk, seeds, res):
                        p = tp[j]
                        text = tok.decode(r["tokens"], skip_special_tokens=True)
                        row = {"model": name, "id": p["id"], "workload": p["workload"], "rep": rep, "seed": s,
                               "n_tokens": len(r["tokens"]), "stop": r["stop"], "truncated": r["stop"] == "cap",
                               "text": text}
                        if dom == "math":
                            ext = D.gsm8k_extract(text)
                            row.update({"extracted": ext, "unextractable": ext is None,
                                        "correct": ext is not None and ext == D._norm(p["answer"])})
                        else:
                            row.update(D.short_score(text, p["answer"]))
                        rows.append(row)
        save_result(OUT / f"accuracy_{name}.jsonl", rows, meta({"model": name, "seconds": time.time() - t0}))
        acc_m = sum(r["correct"] for r in rows if r["workload"] == "math") / 120
        print(f"accuracy {name}: math {acc_m:.3f} ({time.time() - t0:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("distill", "ownbank", "align", "accuracy"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--method", choices=("off", "on"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run")
    ap.add_argument("--models", nargs="+", default=DRAFTS)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    tok = _tok()
    {"distill": distill, "ownbank": ownbank, "align": align, "accuracy": accuracy}[a.mode](a, dev, tok)


if __name__ == "__main__":
    main()
