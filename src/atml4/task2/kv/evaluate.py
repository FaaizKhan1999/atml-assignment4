"""Per-prompt evaluation of the FP16 reference and the five compressed-cache configurations (C5 quality metrics).

For every method: compress the FP16 context cache, verify the serialized bytes (length == formula, deserialize ==
cache tensors bit-exactly), generate greedily (suffix at absolute positions L.., max 16 new tokens, stop on EOS),
extract the answer, teacher-force the gold answer (code digits + <|im_end|>) for the reference-answer NLL, KL of the
first answer-token distribution vs FP16, evidence retained (FP16 selection), K/V reconstruction error, bytes.
Quality only: timings here are NOT clean (see the exclusive timing pass).
"""
from __future__ import annotations

import time

import torch

from . import compress as CP
from . import kvbytes as KB
from .kvcache import RunningQuant, make_cache, prefill, run_suffix_and_generate, teacher_forced_nll
from .prompts import MAX_NEW_TOKENS, Prompt, extract_answer
from .scoring import question_scores


def eos_ids(model, tok) -> list[int]:
    e = model.generation_config.eos_token_id
    e = e if isinstance(e, list) else [e]
    return sorted((set(e) | {tok.convert_tokens_to_ids("<|im_end|>")}) - {None})


def kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    lp, lq = torch.log_softmax(p_logits.float(), -1), torch.log_softmax(q_logits.float(), -1)
    return float((lp.exp() * (lp - lq)).sum())


def _after(c: CP.Compressed):
    return RunningQuant(c.running["k_layout"], c.running["bits"], c.running["n_quantized"]) if c.running else None


@torch.no_grad()
def evaluate_prompt(model, tok, p: Prompt, device, methods=CP.METHODS, verify_serialization: bool = True) -> dict:
    cfg = model.config
    C = KB.channels(cfg)
    eos = eos_ids(model, tok)
    t0 = time.perf_counter()
    Ks, Vs, _ = prefill(model, p.context_ids, device)
    t_prefill = time.perf_counter() - t0
    t0 = time.perf_counter()
    sc = question_scores(model, Ks, Vs, p.suffix_ids, device)
    t_score = time.perf_counter() - t0
    L = p.L
    ev = torch.tensor(p.evidence_tokens)
    rec = {"id": p.id, "cell": [p.case["target_tokens"], p.case["position"], p.case["distractors"]], "L": L,
           "suffix_len": len(p.suffix_ids), "total_len": len(p.ids), "n_evidence_tokens": len(p.evidence_tokens),
           "evidence_token_range": [min(p.evidence_tokens), max(p.evidence_tokens) + 1],
           "code": p.case["code"], "budget_M4": KB.m_b(L, C, 4), "M16": KB.m16(L, C),
           "untimed_seconds": {"prefill": t_prefill, "score": t_score}, "methods": {}}
    ref_first = None
    for m in methods:
        t0 = time.perf_counter()
        c = CP.compress(m, Ks, Vs, cfg, scores=sc["scores"] if m in ("qaware", "hybrid") else None)
        t_comp = time.perf_counter() - t0
        assert c.nbytes == c.formula_bytes, (m, c.nbytes, c.formula_bytes)
        assert c.nbytes <= KB.m_b(L, C, 4) or m == "fp16"
        if verify_serialization:
            kk, vv = CP.deserialize(c, cfg, L)
            assert all(torch.equal(a, b) for a, b in zip(kk, c.keys)) and \
                all(torch.equal(a, b) for a, b in zip(vv, c.values)), f"{m}: deserialized cache differs"
        cache = make_cache(c.keys, c.values)
        P0 = cache.get_seq_length()
        run = _after(c)
        g = run_suffix_and_generate(model, cache, p.suffix_ids, L, device, MAX_NEW_TOKENS, eos, after=run)
        P_end = cache.get_seq_length()
        text = tok.decode(g["generated"], skip_special_tokens=True)
        ans = extract_answer(text)
        tf = teacher_forced_nll(model, make_cache(c.keys, c.values), p.suffix_ids, p.answer_ids, L, device,
                                after=_after(c))
        if m == "fp16":
            ref_first = g["first_logits"]
        appended = P_end - P0
        if c.running:
            live = KB.live_quantized_bytes(run.n_q, P_end - run.n_q, C, 4)
        else:
            live = c.nbytes + 2 * KB.FP16 * C * appended
        sel = c.fp16_selected.cpu() if c.fp16_selected is not None else None
        r = {"generated_ids": g["generated"], "text": text, "answer": ans, "correct": ans == p.case["code"],
             "all_finite": g["all_finite"], "ref_nll_sum": tf["nll_sum"], "ref_nll_per_token": tf["per_token"],
             "ref_nll_tokens": tf["n_tokens"],
             "kl_first_token_vs_fp16": kl(ref_first, g["first_logits"]) if ref_first is not None else None,
             "first_token_argmax": int(g["first_logits"].argmax()),
             "packed_bytes": c.nbytes, "formula_bytes": c.formula_bytes, "fraction_of_M16": c.nbytes / KB.m16(L, C),
             "fraction_of_M4": c.nbytes / KB.m_b(L, C, 4), "breakdown": c.breakdown,
             "physical_len_context": P0, "appended_entries": appended,
             "shared_decode_fp16_bytes": 2 * KB.FP16 * C * appended, "total_live_bytes_end": live,
             "evidence_retained_fp16": float(torch.isin(ev, sel).float().mean()) if sel is not None else None,
             "recon_error": CP.reconstruction_error(c, Ks, Vs), "untimed_compress_s": t_comp}
        if c.running:
            r["groups_quantized_during_decode"] = run.groups_quantized_during_decode
        if m in ("sink_recent", "qaware", "hybrid"):
            r["fp16_selection_rle"] = rle(sel, L)
        rec["methods"][m] = r
        del cache, c
    rec["score_top10"] = torch.topk(sc["scores"], 10).indices.tolist()
    return rec


def rle(sel: torch.Tensor, L: int) -> list[list[int]]:
    """Run-length encoding of the selection mask as [start, length] runs of selected positions."""
    s = sorted(sel.tolist())
    runs = []
    for x in s:
        if runs and runs[-1][0] + runs[-1][1] == x:
            runs[-1][1] += 1
        else:
            runs.append([x, 1])
    return runs
