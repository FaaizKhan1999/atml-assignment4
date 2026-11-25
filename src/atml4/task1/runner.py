"""Shared Task 1 evaluation harness. Every condition is evaluated by exactly this code.

python scripts/task1_eval.py --device cuda:0 --conditions baseline [--phases ...] [--hellaswag-n 2000]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from atml4.common.io import save_result
from atml4.common.meta import collect_meta, model_revision, repo_root
from atml4.common.seed import set_seed

from . import calib, data, hellaswag, ppl, resources, storage
from .conditions import CONDITIONS
from .model import Pristine, eligible_projections, load_base_model, tie_info

PHASES = ("storage", "calib", "ppl", "hellaswag", "timing")
SEED = 6304


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--hellaswag-n", type=int, default=2000, help="2000 (full) or 512 (diagnostic subset)")
    ap.add_argument("--calib-batch", type=int, default=8)
    ap.add_argument("--timing-reps", type=int, default=5)
    ap.add_argument("--timing-exclusive", choices=["yes", "no"], default="yes",
                    help="'yes' only if no other job uses any GPU of this machine during timing (CLAUDE.md)")
    ap.add_argument("--out", default="results/task1")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    phases = [p for p in a.phases.split(",") if p]
    bad = set(phases) - set(PHASES)
    if bad:
        raise SystemExit(f"unknown phases {bad}")
    if "timing" in phases and a.timing_exclusive != "yes":
        raise SystemExit("refusing to time while another GPU job may be running (CLAUDE.md timing rule)")
    for c in a.conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c}; known: {sorted(CONDITIONS)}")
    set_seed(SEED, deterministic=False)
    root = repo_root()
    out_root = root / a.out
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    phase_s: dict[str, float] = {}

    t = time.perf_counter()
    model, tok = load_base_model(a.device)
    torch.cuda.synchronize(dev)
    phase_s["load"] = time.perf_counter() - t
    t = time.perf_counter()
    pristine = Pristine(model)
    phase_s["pristine_snapshot"] = time.perf_counter() - t
    projs = eligible_projections(model)
    sdpa = resources.sdpa_backend_in_model(model, dev)
    mid, rev = model_revision("base")
    common_extra = {
        "tokenizer": {"id": mid, "revision": rev, "class": type(tok).__name__, "vocab_size": tok.vocab_size,
                      "len": len(tok), "bos_token": tok.bos_token, "add_bos": "none (Qwen2 tokenizer)"},
        "device": a.device, "attention_backend": sdpa, "precision": "fp16 weights/activations; fp32 log-softmax",
        "dataset_selection": data.selection_record(["calibration", "wikitext2", "c4", "hellaswag"]),
        "eligible_projections": {"count": len(projs), "num_layers": model.config.num_hidden_layers,
                                 "names": list(dict.fromkeys(p.name for p in projs))},
        "tie": tie_info(model), "pristine_sha256": pristine.sha256,
        "timing_exclusive": a.timing_exclusive == "yes",
    }

    timing_inputs = None
    if "timing" in phases:
        tpath = out_root / "timing_inputs.json"
        timing_inputs = resources.load_or_create_timing_inputs(tpath, tok)
        if not tpath.exists():
            save_result(tpath, timing_inputs, collect_meta(seed=SEED, models=["base"]))

    for cond in a.conditions:
        cphase = dict(phase_s)
        t = time.perf_counter()
        pristine.restore(verify=True)
        cphase["restore_verify"] = time.perf_counter() - t
        t = time.perf_counter()
        cinfo = CONDITIONS[cond](model, {"tokenizer": tok, "device": dev, "pristine": pristine})
        torch.cuda.synchronize(dev)
        cphase["compress"] = time.perf_counter() - t
        meta = collect_meta(seed=SEED, dtype=torch.float16, models=["base"],
                            extra={**common_extra, "condition": cond, "condition_info": cinfo})
        od = out_root / cond
        summary = {"condition": cond, "condition_info": cinfo, "phases_run": phases}

        if "storage" in phases:
            st, per = storage.model_storage_from_torch(model)
            sp = storage.sparse16_model_bytes([(x["numel"], x["nnz"]) for x in per], st.other_bytes)
            nnz = sum(x["nnz"] for x in per)
            summary["storage"] = {"dense": st.to_dict(), "sparse16_estimate": sp,
                                  "nonzero_fraction": nnz / st.eligible_params,
                                  "nominal_bits_per_eligible_weight": 16,
                                  "kind": "logical/serialized bytes (not GPU-allocated)"}
            summary["per_matrix_nnz"] = per

        if "calib" in phases:
            t = time.perf_counter()
            recs = data.load("calibration")
            stats = calib.collect_input_stats(model, calib.calibration_batches(recs, a.calib_batch, dev))
            torch.cuda.synchronize(dev)
            cphase["calib_stats"] = time.perf_counter() - t
            od.mkdir(parents=True, exist_ok=True)
            cpath = od / "calib_input_stats.npz"
            calib.save_stats(cpath, stats, [r["source_row"] for r in recs], SEED)
            summary["calib_stats"] = {"file": str(cpath.relative_to(root)), "bytes": cpath.stat().st_size,
                                      "sequences": len(recs), "tokens_per_seq": len(recs[0]["token_ids"]),
                                      "batch_size": a.calib_batch, "accumulation": "fp32",
                                      "stats": ["sumsq", "sumabs", "count"]}

        if "ppl" in phases:
            for suite, fname in (("wikitext2", "ppl_wikitext2.jsonl"), ("c4", "ppl_c4.jsonl")):
                t = time.perf_counter()
                rows, s = ppl.evaluate_windows(model, data.load_windows(suite), dev)
                torch.cuda.synchronize(dev)
                cphase[f"ppl_{suite}"] = time.perf_counter() - t
                if s["N"] != 50000:
                    raise RuntimeError(f"{suite}: N={s['N']} != 50000")
                save_result(od / fname, rows, meta)
                summary[f"ppl_{suite}"] = s

        if "hellaswag" in phases:
            t = time.perf_counter()
            recs = data.load("hellaswag")[: a.hellaswag_n]
            rows = []
            for i, r in enumerate(recs):
                rows.append(hellaswag.score_example(model, tok, r, dev))
                if (i + 1) % 250 == 0:
                    print(f"  hellaswag {i + 1}/{len(recs)}", flush=True)
            torch.cuda.synchronize(dev)
            cphase["hellaswag"] = time.perf_counter() - t
            hmeta = {**meta, "hellaswag": {"lm_eval_version_followed": hellaswag.LM_EVAL_VERSION,
                                           "task_version": hellaswag.TASK_VERSION,
                                           "target_delimiter": hellaswag.TARGET_DELIMITER,
                                           "prompt_format": hellaswag.PROMPT_FORMAT, "num_fewshot": 0}}
            save_result(od / "hellaswag.jsonl", rows, hmeta)
            summary["hellaswag"] = {"all": hellaswag.summarize(rows)}
            if len(rows) >= data.HELLASWAG_DIAGNOSTIC_N:
                summary["hellaswag"]["first_512"] = hellaswag.summarize(rows, data.HELLASWAG_DIAGNOSTIC_N)

        if "timing" in phases:
            t = time.perf_counter()
            summary["latency_memory"] = resources.measure(model, timing_inputs["token_ids"], dev,
                                                          reps=a.timing_reps)
            cphase["timing"] = time.perf_counter() - t

        summary["phase_seconds"] = cphase
        save_result(od / "summary.json", summary, meta)
        print(f"[{cond}] done: " + ", ".join(f"{k}={v:.1f}s" for k, v in cphase.items()), flush=True)
    pristine.restore(verify=True)
