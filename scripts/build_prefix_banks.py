#!/usr/bin/env python3
"""Build the Task 3 teacher / original-draft prefix banks with the course setup_data.py logic.

The course source (external/atml-pa4-datasets/scripts/setup_data.py) is read and executed UNCHANGED except for
one patch at the generate() call (deviation D1 in docs/deviations.md): `generator=gen` is dropped and the global
torch CPU+CUDA RNGs are seeded with gen.initial_seed() (the exact course per-sequence seed) right before each
generate(). The submodule is never modified on disk.

Flow (one kernel):
  1. smoke: --limit-prefix-prompts 2 equivalent (timed; load and generate() calls instrumented)
  2. extrapolate full 120-prompt runtime; if <= --max-hours continue, else exit(3)
  3. full build -> external/atml-pa4-datasets/data/prefix_banks/prefix_banks_locked.jsonl.gz + manifest
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
import sys
import time
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA_REPO = ROOT / "external/atml-pa4-datasets"
COURSE_SRC = DATA_REPO / "scripts/setup_data.py"
N_FULL_PROMPTS = 120

OLD = """            with torch.inference_mode():
                seq=model.generate(ids,max_new_tokens=192,do_sample=True,temperature=1.0,top_p=1.0,top_k=0,
                     generator=gen,pad_token_id=tok.eos_token_id)[0,ids.shape[1]:].tolist()
"""
NEW = """            torch.manual_seed(gen.initial_seed()); torch.cuda.manual_seed_all(gen.initial_seed())
            with torch.inference_mode():
                seq=model.generate(ids,max_new_tokens=192,do_sample=True,temperature=1.0,top_p=1.0,top_k=0,
                     pad_token_id=tok.eos_token_id)[0,ids.shape[1]:].tolist()
"""


def patched_source() -> tuple[str, str]:
    src = COURSE_SRC.read_text()
    if src.count(OLD) != 1:
        raise SystemExit("course setup_data.py does not contain the expected generate() block exactly once")
    new = src.replace(OLD, NEW)
    diff = "".join(difflib.unified_diff(src.splitlines(True), new.splitlines(True),
                                        "a/scripts/setup_data.py", "b/scripts/setup_data.py"))
    return new, diff


def load_patched_module():
    src, diff = patched_source()
    mod = types.ModuleType("setup_data_patched")
    mod.__file__ = str(COURSE_SRC)  # so ROOT/LOCK/BANK resolve inside the submodule exactly as the course code
    exec(compile(src, str(COURSE_SRC), "exec"), mod.__dict__)
    return mod, diff


class Instrument:
    """Wall-clock timing of model loads and generate() calls (does not change any arguments)."""

    def __init__(self):
        self.loads, self.gens = [], []

    def install(self):
        import torch
        import transformers
        from transformers.generation.utils import GenerationMixin

        inst = self
        orig_fp = transformers.AutoModelForCausalLM.from_pretrained.__func__
        orig_gen = GenerationMixin.generate

        def fp(cls, *a, **k):
            t = time.perf_counter()
            m = orig_fp(cls, *a, **k)
            inst.loads.append({"model": a[0] if a else k.get("pretrained_model_name_or_path"),
                               "s": time.perf_counter() - t})
            return m

        def gen(self, *a, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = orig_gen(self, *a, **k)
            torch.cuda.synchronize()
            inp = a[0] if a else k.get("input_ids")
            inst.gens.append({"model": self.config._name_or_path, "s": time.perf_counter() - t,
                              "new_tokens": int(out.shape[1] - inp.shape[1])})
            return out

        transformers.AutoModelForCausalLM.from_pretrained = classmethod(fp)
        GenerationMixin.generate = gen


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def estimate(inst: Instrument, n_smoke: int) -> dict:
    per_model = {}
    for g in inst.gens:
        d = per_model.setdefault(g["model"], {"gen_s": [], "tokens": []})
        d["gen_s"].append(g["s"])
        d["tokens"].append(g["new_tokens"])
    load_s = sum(x["s"] for x in inst.loads)
    gen_mean = {m: sum(v["gen_s"]) / len(v["gen_s"]) for m, v in per_model.items()}
    full = load_s + sum(N_FULL_PROMPTS * s for s in gen_mean.values())
    return {"smoke_prompts": n_smoke, "loads": inst.loads, "per_model": per_model,
            "mean_generate_s": gen_mean, "load_s_total": load_s, "full_estimate_s": full,
            "full_estimate_h": full / 3600,
            "basis": "load time + 120 x mean generate() time per model from the smoke prompts (GSM8K math, "
                     "the longest responses); short-form prompts likely end earlier, so closer to an upper bound"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-prompts", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=2.5)
    ap.add_argument("--smoke-only", action="store_true")
    a = ap.parse_args()

    from atml4.common.io import save_result
    from atml4.common.meta import collect_meta

    mod, diff = load_patched_module()
    print(diff, flush=True)
    inst = Instrument()
    inst.install()
    rev = mod.revisions(False)  # uses the data-prep lock (data/local_source_revisions.json); never re-resolves
    dest = mod.BANK / "prefix_banks_locked.jsonl.gz"

    t0 = time.time()
    mod.prefix_banks(rev, True, a.smoke_prompts)
    smoke_s = time.time() - t0
    smoke_n = sum(1 for _ in gzip.open(dest, "rt"))
    dest.unlink()  # smoke bank must never be mistaken for the evaluation bank
    est = estimate(inst, a.smoke_prompts)
    est.update({"smoke_wall_s": smoke_s, "smoke_records": smoke_n})
    print("SMOKE ELAPSED_S", round(smoke_s, 1), "ESTIMATE", json.dumps(est), flush=True)

    meta = collect_meta(seed=6304, models=["target", "draft"],
                        extra={"course_setup_data_sha256": sha256(COURSE_SRC), "patch_diff": diff,
                               "data_lock": rev})
    out = {"smoke": est, "max_hours": a.max_hours}
    if a.smoke_only or est["full_estimate_h"] > a.max_hours:
        out["full_build"] = "skipped: " + ("--smoke-only" if a.smoke_only else
                                           f"estimate {est['full_estimate_h']:.2f} h > {a.max_hours} h")
        save_result(ROOT / "results/task3/prefix_banks_manifest.json", out, meta)
        print(out["full_build"], flush=True)
        sys.exit(0 if a.smoke_only else 3)

    inst.loads.clear()
    inst.gens.clear()
    t1 = time.time()
    mod.prefix_banks(rev, True, None)
    full_s = time.time() - t1
    rows = [json.loads(l) for l in gzip.open(dest, "rt")]
    counts = Counter((r["bank"], r["domain"]) for r in rows)
    pos_bins = Counter((r["bank"], "1-16" if r["position"] <= 16 else "17-32" if r["position"] <= 32 else
                        "33-64" if r["position"] <= 64 else "65-192") for r in rows)
    out["full_build"] = {
        "wall_s": full_s, "records": len(rows), "prompts": len({r["prompt_id"] for r in rows}),
        "counts_bank_domain": {f"{b}/{d}": n for (b, d), n in sorted(counts.items())},
        "counts_bank_posbin": {f"{b}/{p}": n for (b, p), n in sorted(pos_bins.items())},
        "generate_calls": len(inst.gens), "generate_s_total": sum(g["s"] for g in inst.gens),
        "loads": inst.loads,
        "files": {str(p.relative_to(DATA_REPO)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in sorted(mod.BANK.glob("*")) if p.is_file()},
    }
    save_result(ROOT / "results/task3/prefix_banks_manifest.json", out, meta)
    (mod.BANK / "prefix_banks_manifest.json").write_text(
        (ROOT / "results/task3/prefix_banks_manifest.json").read_text())
    print("FULL BUILD DONE", json.dumps({k: v for k, v in out["full_build"].items() if k != "loads"}), flush=True)


if __name__ == "__main__":
    main()
