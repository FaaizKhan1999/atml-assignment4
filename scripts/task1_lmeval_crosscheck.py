#!/usr/bin/env python3
"""Cross-check our HellaSwag scoring against lm-evaluation-harness itself on the first --limit of our IDs.

lm-eval's own task object builds the context (fewshot_context, 0-shot), the requests (construct_requests) and the
metrics (process_results); its HFLM wraps the SAME loaded model/tokenizer (batch_size=1). Docs are our locked
raw examples passed through the task's own process_docs. Run twice: softmax_dtype=float32 (our convention) and
lm-eval's default (None = model dtype). Exits 2 if float32 predictions disagree or max |dll| > --tol.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task1 import data, hellaswag  # noqa: E402
from atml4.task1.model import load_base_model  # noqa: E402


def run_lmeval(task, lm_cls, model, tok, docs, softmax_dtype):
    lm = lm_cls(pretrained=model, tokenizer=tok, batch_size=1, softmax_dtype=softmax_dtype)
    insts, ctxs = [], []
    for i, doc in enumerate(docs):
        ctx = task.fewshot_context(doc, num_fewshot=0)
        ctxs.append(ctx)
        reqs = task.construct_requests(doc=doc, ctx=ctx, metadata=("hellaswag", i, 1))
        insts.append(reqs if isinstance(reqs, list) else [reqs])
    flat = [r for g in insts for r in g]
    res = lm.loglikelihood(flat)
    out, k = [], 0
    for doc, g, ctx in zip(docs, insts, ctxs):
        rr = res[k:k + len(g)]
        k += len(g)
        m = task.process_results(doc, rr)
        out.append({"ctx": ctx, "args": [list(r.arguments) for r in g], "loglik": [float(x[0]) for x in rr],
                    "acc": float(m["acc"]), "acc_norm": float(m["acc_norm"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    import datasets
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict

    t0 = time.time()
    model, tok = load_base_model(a.device)
    task = get_task_dict(["hellaswag"])["hellaswag"]
    recs = data.load("hellaswag")[: a.limit]
    pd = task.config.process_docs
    if not callable(pd):
        from lm_eval.tasks.hellaswag.utils import process_docs as pd
    docs = list(pd(datasets.Dataset.from_list([r["example"] for r in recs])))
    ours = [hellaswag.score_example(model, tok, r, torch.device(a.device)) for r in recs]

    report = {"lm_eval_version": lm_eval.__version__, "task_version": task.VERSION if hasattr(task, "VERSION")
              else task.config.metadata, "n": len(recs), "tol": a.tol, "runs": {}}
    fmt_mismatch = []
    for i, (doc, r) in enumerate(zip(docs, recs)):
        d = hellaswag.process_doc(r["example"])
        if d["query"] != doc["query"] or d["choices"] != doc["choices"] or d["gold"] != int(doc["gold"]):
            fmt_mismatch.append(i)
    report["process_doc_mismatches"] = fmt_mismatch

    ok = not fmt_mismatch
    for name, sd in (("softmax_float32", "float32"), ("softmax_default_model_dtype", None)):
        le = run_lmeval(task, HFLM, model, tok, docs, sd)
        diffs = [abs(x - y) for o, l in zip(ours, le) for x, y in zip(o["loglik"], l["loglik"])]
        ctx_eq = all(l["ctx"] == hellaswag.process_doc(r["example"])["query"] for l, r in zip(le, recs))
        args_eq = all(l["args"] == [[hellaswag.process_doc(r["example"])["query"],
                                     hellaswag.TARGET_DELIMITER + c] for c in
                                    hellaswag.process_doc(r["example"])["choices"]] for l, r in zip(le, recs))
        acc_agree = sum(int(o["acc"] == l["acc"]) for o, l in zip(ours, le))
        accn_agree = sum(int(o["acc_norm"] == l["acc_norm"]) for o, l in zip(ours, le))
        run = {"max_abs_loglik_diff": max(diffs), "mean_abs_loglik_diff": sum(diffs) / len(diffs),
               "context_strings_equal": ctx_eq, "request_args_equal": args_eq,
               "acc_agree": acc_agree, "acc_norm_agree": accn_agree,
               "acc_ours": sum(o["acc"] for o in ours) / len(ours),
               "acc_lmeval": sum(l["acc"] for l in le) / len(le),
               "acc_norm_ours": sum(o["acc_norm"] for o in ours) / len(ours),
               "acc_norm_lmeval": sum(l["acc_norm"] for l in le) / len(le),
               "per_example": [{"order": o["order"], "ours": o["loglik"], "lmeval": l["loglik"]}
                               for o, l in zip(ours, le)]}
        report["runs"][name] = run
        if name == "softmax_float32":
            ok = ok and ctx_eq and args_eq and acc_agree == len(ours) and accn_agree == len(ours) \
                and run["max_abs_loglik_diff"] <= a.tol
    report["agree"] = ok
    report["seconds"] = time.time() - t0
    save_result(ROOT / "results/task1/baseline/lmeval_crosscheck.json", report,
                collect_meta(seed=6304, dtype=torch.float16, models=["base"]))
    print({k: (v if k != "runs" else {n: {kk: vv for kk, vv in r.items() if kk != "per_example"}
                                       for n, r in v.items()}) for k, v in report.items()}, flush=True)
    if not ok:
        print("CROSS-CHECK DISAGREES: stop and report", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
