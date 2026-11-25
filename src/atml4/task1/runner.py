"""Shared Task 1 evaluation harness. Every condition is evaluated by exactly this code.

python scripts/task1_eval.py --device cuda:0 --conditions baseline p1_mag_s50 ... [--phases ...]
Phases: storage, calib, outerr, ppl, hellaswag, timing. Results: results/task1/<condition>/; summary.json is
MERGED across invocations (e.g. a quality pass and a separate exclusive timing pass).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import torch

from atml4.common.io import load_result, save_result
from atml4.common.meta import collect_meta, model_revision, repo_root
from atml4.common.seed import set_seed

from . import calib, data, hellaswag, ppl, resources, storage
from .conditions import CONDITIONS, FAMILY, HELLASWAG_N
from .model import Pristine, eligible_projections, load_base_model, tie_info

PHASES = ("storage", "calib", "outerr", "ppl", "hellaswag", "timing")
SEED = 6304


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--phases", default="storage,calib,ppl,hellaswag,timing")
    ap.add_argument("--hellaswag-n", default="auto", help="'auto' = condition default (2000 or 512), or an int")
    ap.add_argument("--calib-batch", type=int, default=8)
    ap.add_argument("--timing-reps", type=int, default=5)
    ap.add_argument("--timing-exclusive", choices=["yes", "no"], default="yes",
                    help="'yes' only if no other job uses any GPU of this machine during timing (CLAUDE.md)")
    ap.add_argument("--no-verify-pack", action="store_true", help="skip pack/unpack + rescaling verification")
    ap.add_argument("--out", default="results/task1")
    return ap.parse_args(argv)


def _merge_summary(path, new: dict, meta: dict) -> None:
    if path.exists():
        old_meta, old = load_result(path)
        merged = dict(old)
        merged["phase_seconds"] = {**old.get("phase_seconds", {}), **new.get("phase_seconds", {})}
        for k, v in new.items():
            if k != "phase_seconds":
                merged[k] = v
        merged["phases_run"] = sorted(set(old.get("phases_run", [])) | set(new.get("phases_run", [])))
        merged["previous_meta"] = old.get("previous_meta", []) + [
            {"command": old_meta.get("command"), "timestamp_utc": old_meta.get("timestamp_utc"),
             "git": old_meta.get("git"), "phases": old.get("phases_run")}]
        new = merged
    save_result(path, new, meta)


def _output_error(model, pristine, recs, batch, dev, equivalent=None) -> dict:
    """Isolated relative output error of every compressed projection (dense inputs from pristine model).
    equivalent(qualname, W') maps folded weights (Q2 down/up) to the operator they replace."""
    projs = eligible_projections(model)
    Wc = {p.qualname: p.module.weight.detach().clone() for p in projs}
    eq = equivalent or (lambda q, w: w)
    pristine.restore(verify=False)
    try:
        err = calib.relative_output_error(model, calib.calibration_batches(recs, batch, dev),
                                          lambda q: eq(q, Wc[q]))
    finally:
        with torch.no_grad():
            for p in projs:
                p.module.weight.copy_(Wc[p.qualname])
        del Wc
        torch.cuda.empty_cache()
    numel = {p.qualname: p.module.weight.numel() for p in projs}
    tot = sum(numel.values())
    return {"per_projection": [{"qualname": q, "rel": v["rel"], "num": v["num"], "den": v["den"]}
                               for q, v in err.items()],
            "param_weighted_mean_rel": sum(err[q]["rel"] * numel[q] for q in err) / tot,
            "mean_rel": sum(v["rel"] for v in err.values()) / len(err),
            "kind": "isolated per-projection ||W X - W' X||_F^2 / ||W X||_F^2 on 128x512 calibration tokens "
                    "(inputs from the unchanged model, one compressed projection at a time)"}


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
        "timing_exclusive": a.timing_exclusive == "yes" if "timing" in phases else None,
        "phases": phases,
    }

    timing_inputs = None
    if "timing" in phases:
        tpath = out_root / "timing_inputs.json"
        timing_inputs = resources.load_or_create_timing_inputs(tpath, tok)
        if not tpath.exists():
            save_result(tpath, timing_inputs, collect_meta(seed=SEED, models=["base"]))
    calib_recs = data.load("calibration") if ("calib" in phases or "outerr" in phases) else None
    failures = []

    for cond in a.conditions:
        cphase = dict(phase_s) if cond == a.conditions[0] else {}
        od = out_root / cond
        try:
            t = time.perf_counter()
            pristine.restore(verify=True)
            cphase["restore_verify"] = time.perf_counter() - t
            t = time.perf_counter()
            ctx = {"tokenizer": tok, "device": dev, "pristine": pristine, "root": root,
                   "verify_pack": not a.no_verify_pack and phases != ["timing"]}
            cinfo = CONDITIONS[cond](model, ctx)
            torch.cuda.synchronize(dev)
            cphase["compress_total"] = time.perf_counter() - t
            meta = collect_meta(seed=SEED, dtype=torch.float16, models=["base"],
                                extra={**common_extra, "condition": cond, "family": FAMILY[cond]})
            summary = {"condition": cond, "family": FAMILY[cond], "phases_run": phases}
            if phases != ["timing"]:
                summary["condition_info"] = cinfo

            if "storage" in phases:
                st, per = storage.model_storage_from_torch(model)
                sp = storage.sparse16_model_bytes([(x["numel"], x["nnz"]) for x in per], st.other_bytes)
                nnz = sum(x["nnz"] for x in per)
                s_out = {"dense": st.to_dict(), "sparse16_estimate": sp,
                         "actual_zero_fraction": 1 - nnz / st.eligible_params,
                         "nonzero_fraction": nnz / st.eligible_params,
                         "kind": "logical/serialized bytes (not GPU-allocated)"}
                if cinfo.get("family") == "prune":
                    s_out["mask_pruned_fraction"] = cinfo["mask_pruned_fraction"]
                    s_out["nominal_bits_per_eligible_weight"] = 16
                    if "storage_two_four_estimate" in cinfo:
                        tf = cinfo["storage_two_four_estimate"]
                        s_out["two_four_estimate"] = {**tf, "total_bytes": tf["eligible_bytes"] + st.other_bytes}
                elif cinfo.get("family") == "quant":
                    pk = cinfo["packing"]
                    elig = pk["packed_bytes"]
                    ser = cinfo.get("serialized_scale_bytes", 0)
                    s_out["quantized"] = {
                        "bits": cinfo["bits"], "nominal_bits_per_eligible_weight": cinfo["bits"],
                        "eligible_packed_bytes": elig, "eligible_formula_bytes": pk["formula_bytes"],
                        "codes_bytes_logical": st.eligible_params * cinfo["bits"] / 8,
                        "zero_point_bytes_logical": st.eligible_params / 128 * cinfo["bits"] / 8,
                        "scale_bytes_logical": st.eligible_params / 128 * 2,
                        "serialized_awq_scale_bytes": ser, "other_params_bytes": st.other_bytes,
                        "total_bytes": elig + ser + st.other_bytes,
                        "effective_bits_per_eligible_weight": 8 * (elig + ser) / st.eligible_params,
                        "pack_roundtrip_exact": pk["roundtrip_exact"], "degenerate_groups": pk["degenerate_groups"],
                        "pad_used": pk["pad_used"]}
                else:
                    s_out["nominal_bits_per_eligible_weight"] = 16
                summary["storage"] = s_out
                summary["per_matrix_nnz"] = per

            if "calib" in phases:
                t = time.perf_counter()
                stats = calib.collect_input_stats(model, calib.calibration_batches(calib_recs, a.calib_batch, dev))
                torch.cuda.synchronize(dev)
                cphase["calib_stats"] = time.perf_counter() - t
                od.mkdir(parents=True, exist_ok=True)
                cpath = od / "calib_input_stats.npz"
                calib.save_stats(cpath, stats, [r["source_row"] for r in calib_recs], SEED)
                summary["calib_stats"] = {"file": str(cpath.relative_to(root)), "bytes": cpath.stat().st_size,
                                          "sequences": len(calib_recs),
                                          "tokens_per_seq": len(calib_recs[0]["token_ids"]),
                                          "batch_size": a.calib_batch, "accumulation": "fp32",
                                          "stats": ["sumsq", "sumabs", "count"]}

            if "outerr" in phases and cond != "baseline":
                t = time.perf_counter()
                summary["output_error"] = _output_error(model, pristine, calib_recs, a.calib_batch, dev,
                                                        ctx.get("outerr_equivalent"))
                if ctx.get("outerr_equivalent"):
                    summary["output_error"]["folded_pairs"] = "down/up measured via unfolded equivalents"
                torch.cuda.synchronize(dev)
                cphase["output_error"] = time.perf_counter() - t

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
                n = HELLASWAG_N[cond] if a.hellaswag_n == "auto" else int(a.hellaswag_n)
                t = time.perf_counter()
                recs = data.load("hellaswag")[:n]
                rows = []
                for i, r in enumerate(recs):
                    rows.append(hellaswag.score_example(model, tok, r, dev))
                    if (i + 1) % 500 == 0:
                        print(f"  [{cond}] hellaswag {i + 1}/{len(recs)}", flush=True)
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
                summary["latency_memory"]["execution"] = "dense FP16 kernels (pruned zeros / fake-quant weights)"
                cphase["timing"] = time.perf_counter() - t

            summary["phase_seconds"] = cphase
            _merge_summary(od / "summary.json", summary, meta)
            print(f"[{cond}] done: " + ", ".join(f"{k}={v:.1f}s" for k, v in cphase.items()), flush=True)
        except Exception as e:  # record and continue with the next condition
            tb = traceback.format_exc()
            print(f"[{cond}] FAILED: {e}\n{tb}", flush=True)
            failures.append({"condition": cond, "error": repr(e), "traceback": tb, "phases": phases})
            od.mkdir(parents=True, exist_ok=True)
            (od / f"FAILED_{'_'.join(phases)}.json").write_text(json.dumps(failures[-1], indent=1))
            torch.cuda.empty_cache()
    pristine.restore(verify=True)
    if failures:
        raise SystemExit(f"{len(failures)} condition(s) failed: {[f['condition'] for f in failures]}")
