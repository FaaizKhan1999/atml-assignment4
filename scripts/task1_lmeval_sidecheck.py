#!/usr/bin/env python3
"""Explain the lm-eval vs harness HellaSwag log-likelihood gap on the first --limit examples.

Variants of OUR scoring: fp32 log-softmax / use_cache=False (primary); (a) FP16 log-softmax; (b) use_cache=True
(model(inp) default call as in lm-eval); (c) primary re-run AFTER HFLM initialisation. lm-eval runs:
softmax_dtype=float32 and default (None). Every variant is compared with both lm-eval runs (max/mean |dLL|).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task1 import data, hellaswag  # noqa: E402
from atml4.task1.model import load_base_model  # noqa: E402


@torch.no_grad()
def score(model, tok, recs, dev, lsm_dtype=torch.float32, use_cache=False):
    out = []
    for r in recs:
        d = hellaswag.process_doc(r["example"])
        lls = []
        for ch in d["choices"]:
            c, k = hellaswag.encode_pair(tok, d["query"], hellaswag.TARGET_DELIMITER + ch)
            inp = torch.tensor((c + k)[:-1], device=dev)[None]
            logits = (model(inp).logits if use_cache else model(inp, use_cache=False).logits)[0, -len(k):]
            lp = torch.log_softmax(logits.to(lsm_dtype), -1)
            lls.append(float(lp.gather(-1, torch.tensor(k, device=dev)[:, None]).double().sum()))
        out.append(lls)
    return out


def cmp(a, b):
    d = [abs(x - y) for p, q in zip(a, b) for x, y in zip(p, q)]
    return {"max_abs": max(d), "mean_abs": sum(d) / len(d), "within_1e-3": max(d) <= 1e-3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import get_task_dict

    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    model, tok = load_base_model(a.device)
    recs = data.load("hellaswag")[: a.limit]
    ours = {"ours_fp32_nocache": score(model, tok, recs, dev),
            "a_ours_fp16_logsoftmax": score(model, tok, recs, dev, lsm_dtype=torch.float16),
            "b_ours_use_cache_true": score(model, tok, recs, dev, use_cache=True)}
    task = get_task_dict(["hellaswag"])["hellaswag"]
    import datasets
    from lm_eval.tasks.hellaswag.utils import process_docs
    docs = list(process_docs(datasets.Dataset.from_list([r["example"] for r in recs])))
    le = {}
    for name, sd in (("lmeval_softmax_float32", "float32"), ("lmeval_default", None)):
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=1, softmax_dtype=sd)
        insts = []
        for i, doc in enumerate(docs):
            ctx = task.fewshot_context(doc, num_fewshot=0)
            insts.append(task.construct_requests(doc=doc, ctx=ctx, metadata=("hellaswag", i, 1)))
        res = lm.loglikelihood([r for g in insts for r in g])
        le[name] = [[float(x[0]) for x in res[4 * i:4 * i + 4]] for i in range(len(docs))]
    ours["c_ours_fp32_nocache_after_hflm_init"] = score(model, tok, recs, dev)
    table = {o: {l: cmp(ours[o], le[l]) for l in le} for o in ours}
    table["lmeval_float32_vs_default"] = cmp(le["lmeval_softmax_float32"], le["lmeval_default"])
    table["ours_before_vs_after_hflm"] = cmp(ours["ours_fp32_nocache"], ours["c_ours_fp32_nocache_after_hflm_init"])
    matched = [o for o in ours if any(table[o][l]["within_1e-3"] for l in le)]
    payload = {"lm_eval_version": lm_eval.__version__, "n": len(recs), "comparisons": table,
               "variants_matching_lmeval_within_1e-3": matched, "loglik": {**ours, **le}}
    save_result(ROOT / "results/task1/baseline/lmeval_sidecheck.json", payload,
                collect_meta(seed=6304, dtype=torch.float16, models=["base"]))
    for k, v in table.items():
        print(k, v, flush=True)
    print("MATCHING:", matched, flush=True)


if __name__ == "__main__":
    main()
