"""Zero-shot HellaSwag scoring following lm-evaluation-harness 0.4.13 `hellaswag` (task version 1.0).

Source followed (lm_eval 0.4.13 wheel):
  * tasks/hellaswag/utils.py  preprocess() and process_docs() -> query / choices / gold   (copied verbatim below)
  * tasks/hellaswag/hellaswag.yaml  doc_to_text "{{query}}", doc_to_choice "choices", output_type multiple_choice
  * config default target_delimiter " "; requests are (ctx, " " + choice)
  * api/model.py _encode_pair(): trailing spaces of ctx moved to the continuation; whole = enc(ctx+cont),
    ctx_enc = enc(ctx), cont_enc = whole[len(ctx_enc):]
  * models/huggingface.py tok_encode(): tokenizer default special tokens (Qwen2 adds no BOS);
    input = (ctx_enc + cont_enc)[:-1]; logits of the last len(cont_enc) positions; log_softmax(dtype=softmax_dtype)
  * api/task.py process_results: acc = argmax(lls); acc_norm = argmax(lls / len(choice)) with len in CHARACTERS
    of the preprocessed choice (without the delimiter).
Our scoring computes log_softmax in FP32 (lm-eval default softmax_dtype=None keeps model dtype).
"""
from __future__ import annotations

import re

import torch

LM_EVAL_VERSION = "0.4.13"
TASK_VERSION = "1.0"
TARGET_DELIMITER = " "
PROMPT_FORMAT = {
    "query": 'preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize())',
    "choice": "preprocess(ending)",
    "context": "{query}",
    "continuation": '" " + {choice}',
    "acc_norm": "argmax(loglik / len(choice))  # characters, no delimiter",
    "preprocess": 'strip; replace(" [title]", ". "); re.sub(r"\\[.*?\\]", ""); replace("  ", " ")',
}


# --- verbatim from lm_eval/tasks/hellaswag/utils.py (lm-eval 0.4.13, MIT license) ---
def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def _process_doc(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    out_doc = {
        "query": preprocess(doc["activity_label"] + ": " + ctx),
        "choices": [preprocess(ending) for ending in doc["endings"]],
        "gold": int(doc["label"]),
    }
    return out_doc
# --- end verbatim ---


def process_doc(doc: dict) -> dict:
    return _process_doc(doc)


def encode_pair(tok, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """lm-eval HFLM causal _encode_pair with tokenizer-default special tokens."""
    assert context, "Context cannot be empty!"
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = tok.encode(context + continuation)
    context_enc = tok.encode(context)
    return context_enc, whole_enc[len(context_enc):]


@torch.no_grad()
def continuation_loglik(model, ctx_enc: list[int], cont_enc: list[int], device) -> tuple[float, bool]:
    inp = torch.tensor((ctx_enc + cont_enc)[:-1], dtype=torch.long, device=device)[None]
    logits = model(inp, use_cache=False).logits[0, -len(cont_enc):]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tgt = torch.tensor(cont_enc, device=device)
    ll = float(logp.gather(-1, tgt[:, None]).double().sum())
    greedy = bool((logp.argmax(-1) == tgt).all())
    return ll, greedy


def score_example(model, tok, rec: dict, device) -> dict:
    """rec: one hellaswag_val_locked record {order, source_row_index, example}."""
    d = process_doc(rec["example"])
    lls, greedy, ntoks = [], [], []
    for ch in d["choices"]:
        c, k = encode_pair(tok, d["query"], TARGET_DELIMITER + ch)
        ll, g = continuation_loglik(model, c, k, device)
        lls.append(ll)
        greedy.append(g)
        ntoks.append(len(k))
    clen = [len(ch) for ch in d["choices"]]
    pred = max(range(4), key=lambda i: lls[i])
    pred_norm = max(range(4), key=lambda i: lls[i] / clen[i])
    return {"order": rec["order"], "id": f"hellaswag-val-{rec['source_row_index']}",
            "row_index": rec["source_row_index"], "gold": d["gold"], "loglik": lls, "char_len": clen,
            "cont_tokens": ntoks, "is_greedy": greedy, "pred_acc": pred, "pred_acc_norm": pred_norm,
            "acc": int(pred == d["gold"]), "acc_norm": int(pred_norm == d["gold"])}


def summarize(rows: list[dict], n: int | None = None) -> dict:
    r = rows[:n] if n else rows
    return {"n": len(r), "acc": sum(x["acc"] for x in r) / len(r), "acc_norm": sum(x["acc_norm"] for x in r) / len(r)}
