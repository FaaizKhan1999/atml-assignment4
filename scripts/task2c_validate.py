#!/usr/bin/env python3
"""Task 2C validation stage (small GPU kernel; quality/mechanics only, no timing claims).

1. Encoded lengths of all 180 prompts (context L, suffix, total) + split checks.
2. C1 on the real model (Qwen2.5-1.5B-Instruct FP16 SDPA) for 6 prompts (first prompt of each length x position,
   0 distractors): (a) direct full forward vs split prefill(context) + forward(suffix) logits; (b) 100%-retention
   edited-cache greedy vs model.generate greedy (16 tokens, EOS stop), token-for-token, logits finite;
   (c) eviction of a sink+recent set then appending the suffix with absolute position_ids vs a full forward with a
   4D mask hiding the evicted positions from the suffix rows.
3. Full-FP16 retrieval accuracy on all 180 prompts (18 cells) via the 100%-retention path.
4. evaluate_prompt (all six configurations, serialization verified) on 3 prompts -> bytes per policy, n_kept,
   hybrid FP16 counts, outputs; per-prompt wall time for the full-run estimate.
Writes results/task2/partC/{lengths.json, validation.json}.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta, model_revision  # noqa: E402
from atml4.common.seed import set_seed  # noqa: E402
from atml4.task2.kv import compress as CP  # noqa: E402
from atml4.task2.kv import kvbytes as KB  # noqa: E402
from atml4.task2.kv import prompts as PR  # noqa: E402
from atml4.task2.kv.evaluate import eos_ids, evaluate_prompt, kl  # noqa: E402
from atml4.task2.kv.kvcache import forward_step, make_cache, prefill, run_suffix_and_generate  # noqa: E402

OUT = ROOT / "results/task2/partC"


def load(device):
    mid, rev = model_revision("target")
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.float16, attn_implementation="sdpa")
    model.to(device).eval().requires_grad_(False)
    return model, tok


@torch.no_grad()
def c1(model, tok, p, device):
    eos = eos_ids(model, tok)
    S = len(p.suffix_ids)
    full = model(torch.tensor([p.ids], device=device), logits_to_keep=S).logits[0].float()
    Ks, Vs, _ = prefill(model, p.context_ids, device)
    split = forward_step(model, make_cache(Ks, Vs), p.suffix_ids, p.L, device)
    a = {"max_abs_logit_diff": float((full - split).abs().max()),
         "max_kl_over_suffix_positions": max(kl(full[i], split[i]) for i in range(S)),
         "argmax_agree_all_suffix_positions": bool((full.argmax(-1) == split.argmax(-1)).all()),
         "max_abs_logit": float(full.abs().max())}
    ref = model.generate(torch.tensor([p.ids], device=device), max_new_tokens=PR.MAX_NEW_TOKENS, do_sample=False,
                         eos_token_id=eos, pad_token_id=eos[0], temperature=None, top_p=None, top_k=None,
                         repetition_penalty=1.0)  # Instruct generation_config sets 1.1; greedy = pure argmax
    ref = ref[0, len(p.ids):].tolist()
    g = run_suffix_and_generate(model, make_cache(Ks, Vs), p.suffix_ids, p.L, device, PR.MAX_NEW_TOKENS, eos)
    b = {"generate_ids": ref, "edited_cache_ids": g["generated"], "token_for_token_equal": ref == g["generated"],
         "all_logits_finite": g["all_finite"], "text": tok.decode(g["generated"], skip_special_tokens=True)}
    # (c) eviction mechanics vs 4D-masked full forward
    C = KB.channels(model.config)
    n = KB.n_kept_max(KB.m_b(p.L, C, 4), C, p.L)
    keep = CP.select_sink_recent(p.L, n, device)
    T = len(p.ids)
    allowed = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    ev = torch.ones(p.L, dtype=torch.bool, device=device)
    ev[keep] = False
    allowed[p.L:, :p.L] &= ~ev[None]
    mask4 = torch.zeros(1, 1, T, T, dtype=torch.float16, device=device).masked_fill(~allowed, torch.finfo(torch.float16).min)
    ref_c = model(torch.tensor([p.ids], device=device), attention_mask=mask4,
                  position_ids=torch.arange(T, device=device)[None], logits_to_keep=S).logits[0].float()
    del mask4
    got = forward_step(model, make_cache([k[:, :, keep] for k in Ks], [v[:, :, keep] for v in Vs]), p.suffix_ids,
                       p.L, device)
    wrong = forward_step(model, make_cache([k[:, :, keep] for k in Ks], [v[:, :, keep] for v in Vs]), p.suffix_ids,
                         n, device)  # deliberately wrong: physical count as RoPE position
    c = {"n_kept": n, "max_abs_logit_diff": float((ref_c - got).abs().max()),
         "max_kl": max(kl(ref_c[i], got[i]) for i in range(S)),
         "argmax_agree": bool((ref_c.argmax(-1) == got.argmax(-1)).all()),
         "control_wrong_positions_max_abs_diff": float((ref_c - wrong).abs().max()),
         "control_wrong_positions_max_kl": max(kl(ref_c[i], wrong[i]) for i in range(S))}
    return {"id": p.id, "L": p.L, "split_vs_full": a, "full_retention_vs_generate": b, "eviction_vs_mask": c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    set_seed(6304, deterministic=False)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    model, tok = load(dev)
    cfg = model.config
    cases = PR.load_cases()
    prompts = [PR.build(c, tok) for c in cases]
    C = KB.channels(cfg)
    lengths = [{"id": p.id, "cell": list(PR.cell(p.case)), "L_context": p.L, "suffix": len(p.suffix_ids),
                "total": len(p.ids), "evidence_tokens": [min(p.evidence_tokens), max(p.evidence_tokens) + 1],
                "M4": KB.m_b(p.L, C, 4), "M16": KB.m16(p.L, C), "n_kept": KB.n_kept_max(KB.m_b(p.L, C, 4), C, p.L),
                "hybrid_f": KB.hybrid_f_max(KB.m_b(p.L, C, 4), p.L, C)} for p in prompts]
    meta = collect_meta(seed=6304, dtype=torch.float16, models=["target"],
                        extra={"config": {"n_layers": cfg.num_hidden_layers, "H": cfg.num_attention_heads,
                                          "H_kv": cfg.num_key_value_heads,
                                          "d_h": getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
                                          "C": C},
                               "attn_implementation": model.config._attn_implementation, "eos_ids": eos_ids(model, tok),
                               "split_rule": PR.__doc__})
    save_result(OUT / "lengths.json", {"prompts": lengths}, meta)
    print("lengths saved", flush=True)
    sel = []
    for T in (2048, 4096):
        for pos in ("start", "middle", "end"):
            sel.append(next(p for p in prompts if PR.cell(p.case) == (T, pos, 0)))
    c1res = []
    for p in sel:
        r = c1(model, tok, p, dev)
        c1res.append(r)
        print("C1", p.id, r["split_vs_full"], r["full_retention_vs_generate"]["token_for_token_equal"],
              {k: v for k, v in r["eviction_vs_mask"].items()}, flush=True)
        torch.cuda.empty_cache()
    # 3) full-FP16 accuracy, all 180 prompts
    eos = eos_ids(model, tok)
    t = time.time()
    fp = []
    for p in prompts:
        Ks, Vs, _ = prefill(model, p.context_ids, dev)
        g = run_suffix_and_generate(model, make_cache(Ks, Vs), p.suffix_ids, p.L, dev, PR.MAX_NEW_TOKENS, eos)
        text = tok.decode(g["generated"], skip_special_tokens=True)
        ans = PR.extract_answer(text)
        fp.append({"id": p.id, "cell": list(PR.cell(p.case)), "text": text, "answer": ans,
                   "correct": ans == p.case["code"], "all_finite": g["all_finite"]})
        del Ks, Vs
    t_fp = time.time() - t
    cells = {}
    for r in fp:
        cells.setdefault(str(tuple(r["cell"])), []).append(r["correct"])
    print("FP16 accuracy", sum(r["correct"] for r in fp) / len(fp), flush=True)
    # 4) all six configurations on 3 prompts
    ev_sel = [prompts[0], next(p for p in prompts if PR.cell(p.case) == (2048, "middle", 8)),
              next(p for p in prompts if PR.cell(p.case) == (4096, "middle", 2))]
    evals, per_prompt_s = [], []
    for p in ev_sel:
        t = time.time()
        r = evaluate_prompt(model, tok, p, dev)
        torch.cuda.synchronize(dev)
        per_prompt_s.append({"id": p.id, "L": p.L, "seconds": time.time() - t})
        evals.append(r)
        print("EVAL", p.id, {m: (x["answer"], x["packed_bytes"], round(x["fraction_of_M16"], 4)) for m, x in r["methods"].items()},
              flush=True)
        torch.cuda.empty_cache()
    payload = {"c1": c1res,
               "fp16_full_cache": {"per_prompt": fp, "accuracy": sum(r["correct"] for r in fp) / len(fp),
                                   "per_cell": {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in cells.items()},
                                   "seconds": t_fp},
               "evaluate_prompt": evals, "evaluate_prompt_wall_seconds": per_prompt_s,
               "total_seconds": time.time() - t_all,
               "note": "untimed / non-exclusive; quality and mechanics only"}
    save_result(OUT / "validation.json", payload, meta)
    print(f"validation done in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
