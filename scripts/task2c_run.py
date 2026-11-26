#!/usr/bin/env python3
"""Task 2C full run.

  quality --shard i --nshards n : evaluate_prompt (FP16 reference + five compressed configurations) on prompts
                                  i, i+n, ... -> results/task2/partC/quality_s<i>.jsonl (non-exclusive; no timing).
  timing                        : EXCLUSIVE timing on the fixed subset = first 2 prompt IDs (manifest order) of each
                                  context length; per configuration >= 10 warmed, synced reps of prefill, scoring
                                  (question-aware / hybrid), preprocessing (selection + quantize + pack +
                                  serialize), decode (suffix + greedy, running-group rule); peak allocated memory of
                                  one untimed end-to-end pass and of decode alone -> timing.json.
  bench                         : EXCLUSIVE packed-cache benchmark: real cached K/V of the first prompt of each
                                  nominal length (actual encoded L) and of its first 2,048 / 4,096 positions,
                                  replicated to batch 8; encode (quantize + pack), decode (unpack + dequantize),
                                  serialized bytes, decode-step latency with unpacking vs FP16 at matched shapes
                                  -> packed_bench.json.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.seed import set_seed  # noqa: E402
from atml4.common.timing import read_peak_memory, reset_peak_memory, time_cuda  # noqa: E402
from atml4.task2.kv import compress as CP  # noqa: E402
from atml4.task2.kv import kvbytes as KB  # noqa: E402
from atml4.task2.kv import kvquant as KQ  # noqa: E402
from atml4.task2.kv import prompts as PR  # noqa: E402
from atml4.task2.kv.evaluate import eos_ids, evaluate_prompt  # noqa: E402
from atml4.task2.kv.model import load  # noqa: E402
from atml4.task2.kv.kvcache import RunningQuant, make_cache, prefill, run_suffix_and_generate  # noqa: E402
from atml4.task2.kv.scoring import question_scores  # noqa: E402

OUT = ROOT / "results/task2/partC"
REPS, WARM = 10, 3


def gpu_meta(model, extra):
    return collect_meta(seed=6304, dtype=torch.float16, models=["target"], extra=extra)


def quality(a, model, tok, prompts, dev):
    rows = []
    out = OUT / f"quality_s{a.shard}.jsonl"
    for i, p in enumerate(prompts):
        if i % a.nshards != a.shard:
            continue
        t = time.time()
        r = evaluate_prompt(model, tok, p, dev)
        r["wall_seconds_untimed"] = time.time() - t
        rows.append(r)
        print(f"[q{a.shard}] {p.id} L={p.L} " + " ".join(f"{m}={'Y' if x['correct'] else 'n'}"
                                                         for m, x in r["methods"].items()), flush=True)
        torch.cuda.empty_cache()
    save_result(out, rows, gpu_meta(model, {"shard": a.shard, "nshards": a.nshards, "exclusive": False}))


def timing(a, model, tok, prompts, dev):
    from atml4.task1.resources import gpu_state

    eos = eos_ids(model, tok)
    subset = []
    for T in (2048, 4096):
        subset += [p for p in prompts if p.case["target_tokens"] == T][:2]
    res = []
    for p in subset:
        Ks, Vs, _ = prefill(model, p.context_ids, dev)
        sc = question_scores(model, Ks, Vs, p.suffix_ids, dev)["scores"]
        for m in CP.METHODS:
            rec = {"id": p.id, "L": p.L, "method": m, "reps": REPS, "warmup": WARM, "unit": "ms"}
            rec["prefill"] = time_cuda(lambda: prefill(model, p.context_ids, dev), reps=REPS, warmup=WARM,
                                       min_reps=10, device=dev).to_dict()
            if m in ("qaware", "hybrid"):
                rec["scoring"] = time_cuda(lambda: question_scores(model, Ks, Vs, p.suffix_ids, dev), reps=REPS,
                                           warmup=WARM, min_reps=10, device=dev).to_dict()
            rec["preprocess_compress_pack"] = time_cuda(
                lambda: CP.compress(m, Ks, Vs, model.config, scores=sc if m in ("qaware", "hybrid") else None),
                reps=REPS, warmup=WARM, min_reps=10, device=dev).to_dict()
            c = CP.compress(m, Ks, Vs, model.config, scores=sc if m in ("qaware", "hybrid") else None)

            def setup():
                return make_cache(c.keys, c.values), (RunningQuant(c.running["k_layout"], c.running["bits"],
                                                                   c.running["n_quantized"]) if c.running else None)
            ntok = []
            rec["decode_suffix_plus_greedy"] = time_cuda(
                lambda s: ntok.append(len(run_suffix_and_generate(model, s[0], p.suffix_ids, p.L, dev,
                                                                  PR.MAX_NEW_TOKENS, eos, after=s[1])["generated"])),
                reps=REPS, warmup=WARM, min_reps=10, device=dev, setup=setup).to_dict()
            rec["generated_tokens"] = sorted(set(ntok))
            # peak allocated: one untimed end-to-end pass; and decode alone (cache already built)
            del c
            torch.cuda.empty_cache()
            reset_peak_memory(dev)
            K2, V2, _ = prefill(model, p.context_ids, dev)
            s2 = question_scores(model, K2, V2, p.suffix_ids, dev)["scores"] if m in ("qaware", "hybrid") else None
            c2 = CP.compress(m, K2, V2, model.config, scores=s2)
            del K2, V2
            cache2 = make_cache(c2.keys, c2.values)
            run = RunningQuant(c2.running["k_layout"], 4, c2.running["n_quantized"]) if c2.running else None
            run_suffix_and_generate(model, cache2, p.suffix_ids, p.L, dev, PR.MAX_NEW_TOKENS, eos, after=run)
            rec["peak_end_to_end"] = read_peak_memory(dev)
            del cache2, c2
            torch.cuda.empty_cache()
            c3 = CP.compress(m, Ks, Vs, model.config, scores=sc if m in ("qaware", "hybrid") else None)
            cache = make_cache(c3.keys, c3.values)
            run3 = RunningQuant(c3.running["k_layout"], 4, c3.running["n_quantized"]) if c3.running else None
            del c3
            reset_peak_memory(dev)
            base = torch.cuda.memory_allocated(dev)
            run_suffix_and_generate(model, cache, p.suffix_ids, p.L, dev, PR.MAX_NEW_TOKENS, eos, after=run3)
            rec["peak_decode_only"] = {**read_peak_memory(dev), "allocated_before_decode_bytes": base}
            del cache
            torch.cuda.empty_cache()
            res.append(rec)
            print(f"[timing] {p.id} {m:12s} prefill={rec['prefill']['median']:.1f} decode="
                  f"{rec['decode_suffix_plus_greedy']['median']:.1f} prep={rec['preprocess_compress_pack']['median']:.1f}"
                  + (f" score={rec['scoring']['median']:.1f}" if "scoring" in rec else ""), flush=True)
        del Ks, Vs
        torch.cuda.empty_cache()
    save_result(OUT / "timing.json", {"rows": res, "subset": [p.id for p in subset]},
                gpu_meta(model, {"timing_exclusive": True, "other_gpu_state": gpu_state()}))


def _encode(Ks, Vs, kind, qpos=None, nq=None):
    """Per-layer packed blobs (GPU). Ks: per layer [B*H, L, d] fp16."""
    out = []
    for k, v in zip(Ks, Vs):
        if kind == "int2_hybrid_low":
            qk, qv = KQ.quantize_tensor(k[:, qpos], "channel", 2), KQ.quantize_tensor(v[:, qpos], "token", 2)
        else:
            qk = KQ.quantize_tensor(k[:, :nq], "channel" if kind == "int4_chan" else "token", 4)
            qv = KQ.quantize_tensor(v[:, :nq], "token", 4)
        out.append(torch.cat(KQ.serialize_quant(qk) + KQ.serialize_quant(qv)))
    return out


def _decode(blobs, shape, kind):
    BH, N, d = shape
    bits = 2 if kind == "int2_hybrid_low" else 4
    kl = "token" if kind == "int4_tok" else "channel"
    out = []
    for b in blobs:
        k, off = KQ.deserialize_quant(b, 0, (BH, N, d), kl, bits)
        v, off = KQ.deserialize_quant(b, off, (BH, N, d), "token", bits)
        out.append((k, v))
    return out


def bench(a, model, tok, prompts, dev):
    from atml4.task1.resources import gpu_state

    cfg = model.config
    H = cfg.num_key_value_heads
    rows = []
    for T in (2048, 4096):
        p = next(q for q in prompts if q.case["target_tokens"] == T)
        Ks0, Vs0, _ = prefill(model, p.context_ids, dev)
        sc = question_scores(model, Ks0, Vs0, p.suffix_ids, dev)["scores"]
        for Ltag, Lb in (("actual", p.L), ("nominal", T)):
            for B in (1, 8):
                Ks = [k[:, :, :Lb].expand(B, -1, -1, -1).reshape(B * H, Lb, -1).contiguous() for k in Ks0]
                Vs = [v[:, :, :Lb].expand(B, -1, -1, -1).reshape(B * H, Lb, -1).contiguous() for v in Vs0]
                d = Ks[0].shape[-1]
                nq = (Lb // 64) * 64
                C = KB.channels(cfg)
                f = KB.hybrid_f_max(KB.m_b(Lb, C, 4), Lb, C)
                sel = CP.select_qaware(sc[:Lb], f)
                mask = torch.zeros(Lb, dtype=torch.bool, device=dev)
                mask[sel] = True
                low = torch.arange(Lb, device=dev)[~mask]
                qpos = low[:(low.numel() // 64) * 64]
                for kind in ("int4_chan", "int4_tok", "int2_hybrid_low"):
                    N = qpos.numel() if kind == "int2_hybrid_low" else nq
                    enc = time_cuda(lambda: _encode(Ks, Vs, kind, qpos, nq), reps=REPS, warmup=WARM, min_reps=10,
                                    device=dev)
                    blobs = _encode(Ks, Vs, kind, qpos, nq)
                    dec = time_cuda(lambda: _decode(blobs, (B * H, N, d), kind), reps=REPS, warmup=WARM, min_reps=10,
                                    device=dev)
                    nbytes = sum(b.numel() for b in blobs)
                    rows.append({"nominal": T, "L_tag": Ltag, "L": Lb, "B": B, "kind": kind, "quantized_tokens": N,
                                 "encode_quantize_pack_ms": enc.to_dict(), "decode_unpack_dequant_ms": dec.to_dict(),
                                 "serialized_quantized_part_bytes": nbytes,
                                 "fp16_same_tokens_bytes": 2 * 2 * len(Ks) * B * H * N * d,
                                 "note": "quantized part only; FP16 residual / FP16 set / indices excluded here"})
                    print(f"[bench] T={T} {Ltag} L={Lb} B={B} {kind}: enc {enc.median:.2f} ms dec {dec.median:.2f} ms "
                          f"bytes {nbytes}", flush=True)
                    del blobs
                # decode-step latency: FP16 cache vs unpack+dequant (INT4 chanK/tokV + FP16 residual) + step
                blobs = _encode(Ks, Vs, "int4_chan", None, nq)
                tokx = torch.full((B, 1), p.suffix_ids[0], device=dev)
                pos = torch.full((B, 1), Lb, device=dev)
                cpos = torch.tensor([Lb], device=dev)
                attn = torch.ones(B, Lb + 1, dtype=torch.long, device=dev)

                def step(cache):
                    model(input_ids=tokx, past_key_values=cache, position_ids=pos, cache_position=cpos,
                          attention_mask=attn, use_cache=True)

                def fp16_setup():
                    return DynamicCache(ddp_cache_data=[(k.view(B, H, Lb, d).clone(), v.view(B, H, Lb, d).clone())
                                                        for k, v in zip(Ks, Vs)])

                t_fp = time_cuda(step, reps=REPS, warmup=WARM, min_reps=10, device=dev, setup=fp16_setup)

                def unpack_step():
                    kv = _decode(blobs, (B * H, nq, d), "int4_chan")
                    cache = DynamicCache(ddp_cache_data=[
                        (torch.cat([k, kr[:, nq:]], 1).view(B, H, Lb, d), torch.cat([v, vr[:, nq:]], 1).view(B, H, Lb, d))
                        for (k, v), kr, vr in zip(kv, Ks, Vs)])
                    step(cache)

                t_up = time_cuda(unpack_step, reps=REPS, warmup=WARM, min_reps=10, device=dev)
                reset_peak_memory(dev)
                unpack_step()
                mem_up = read_peak_memory(dev)
                reset_peak_memory(dev)
                step(fp16_setup())
                mem_fp = read_peak_memory(dev)
                rows.append({"nominal": T, "L_tag": Ltag, "L": Lb, "B": B, "kind": "decode_step",
                             "fp16_cache_step_ms": t_fp.to_dict(), "int4_chan_unpack_dequant_plus_step_ms": t_up.to_dict(),
                             "peak_alloc_fp16": mem_fp, "peak_alloc_unpack": mem_up,
                             "packed_bytes_int4_chan_total": KB.m_b(Lb, KB.channels(cfg), 4) * B,
                             "fp16_bytes_total": KB.m16(Lb, KB.channels(cfg)) * B})
                print(f"[bench] T={T} {Ltag} B={B} step fp16 {t_fp.median:.2f} ms vs unpack+step {t_up.median:.2f} ms",
                      flush=True)
                del blobs, Ks, Vs
                torch.cuda.empty_cache()
        del Ks0, Vs0
        torch.cuda.empty_cache()
    save_result(OUT / "packed_bench.json", {"rows": rows},
                gpu_meta(model, {"timing_exclusive": True, "other_gpu_state": gpu_state()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("quality", "timing", "bench"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    set_seed(6304, deterministic=False)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    OUT.mkdir(parents=True, exist_ok=True)
    model, tok = load(dev)
    prompts = [PR.build(c, tok) for c in PR.load_cases()]
    {"quality": quality, "timing": timing, "bench": bench}[a.mode](a, model, tok, prompts, dev)


if __name__ == "__main__":
    main()
