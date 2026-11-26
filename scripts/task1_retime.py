#!/usr/bin/env python3
"""Clean Task 1 re-timing (exclusive: the other GPU must be idle).

Order: counterbalanced ABBA = forward list then its reverse; the forward list is baseline followed by the other
conditions with a baseline drift reference inserted after every 3 conditions (and at the end). Each measurement:
prefill 512, prefill 2048, 64-token decode after 512, >= 5 warmed reps, SM/memory clock, temperature and power
logged after every rep. Tries `nvidia-smi -lgc` (clock lock) and records whether it is permitted.
Writes results/task1/retiming.json and marks the old `latency_memory` in each summary.json as superseded.
"""
import argparse
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from atml4.common.io import load_result, save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.seed import set_seed  # noqa: E402
from atml4.task1 import resources  # noqa: E402
from atml4.task1.conditions import CONDITIONS, FIXED_CONDITIONS, NEW_CONDITIONS, Q3PROJ_CONDITIONS  # noqa: E402
from atml4.task1.model import Pristine, load_base_model  # noqa: E402

METRICS = ("prefill_512", "prefill_2048", "decode_64_after_512")


def make_sampler(index: int):
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(index)

        def sample():
            return {"sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                    "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                    "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                    "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, "t": time.time()}
        return sample, "pynvml"
    except Exception:
        def sample():
            q = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,"
                                "power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            sm, mem, temp, pw = [x.strip() for x in q.stdout.split(",")]
            return {"sm_mhz": float(sm), "mem_mhz": float(mem), "temp_c": float(temp), "power_w": float(pw),
                    "t": time.time()}
        return sample, "nvidia-smi"


def sh(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": " ".join(cmd), "rc": p.returncode, "out": (p.stdout + p.stderr).strip()[:500]}


def sequence(conds):
    others = [c for c in conds if c != "baseline"]
    fwd = ["baseline"]
    for i, c in enumerate(others, 1):
        fwd.append(c)
        if i % 3 == 0:
            fwd.append("baseline")
    if fwd[-1] != "baseline":
        fwd.append("baseline")
    return fwd + fwd[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--lock-mhz", type=int, default=1305)
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="short pass: these conditions + baseline references (default: all Task 1 conditions)")
    ap.add_argument("--out", default="retiming.json")
    a = ap.parse_args()
    set_seed(6304, deterministic=False)
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    idx = dev.index or 0
    lock = {"attempt": sh(["nvidia-smi", "-i", str(idx), "-lgc", f"{a.lock_mhz},{a.lock_mhz}"])}
    lock["permitted"] = lock["attempt"]["rc"] == 0
    sampler, sampler_kind = make_sampler(idx)
    model, tok = load_base_model(a.device)
    pr = Pristine(model)
    tin = load_result(ROOT / "results/task1/timing_inputs.json")[1]["token_ids"]
    conds = ["baseline"] + (a.conditions if a.conditions else FIXED_CONDITIONS + NEW_CONDITIONS)
    assert all(c in CONDITIONS for c in conds) and Q3PROJ_CONDITIONS
    seq = sequence(conds)
    ctx = {"tokenizer": tok, "device": dev, "pristine": pr, "root": ROOT, "verify_pack": False}
    runs = []
    t0 = time.time()
    for pos, c in enumerate(seq):
        pr.restore(verify=False)
        CONDITIONS[c](model, ctx)
        torch.cuda.synchronize(dev)
        m = resources.measure(model, tin, dev, reps=a.reps, sampler=sampler)
        runs.append({"pos": pos, "pass": "forward" if pos < len(seq) // 2 else "reverse", "condition": c,
                     "is_reference": c == "baseline", **{k: m[k] for k in METRICS},
                     "gpu_state_before": m["gpu_state_before"]})
        print(f"[{pos:02d}] {c:16s} " + " ".join(f"{k}={m[k]['median']:.1f}" for k in METRICS) +
              f" sm={m['prefill_512']['per_rep_gpu'][-1]['sm_mhz']} T={m['prefill_512']['per_rep_gpu'][-1]['temp_c']}",
              flush=True)
    pr.restore(verify=True)
    if lock["permitted"]:
        lock["reset"] = sh(["nvidia-smi", "-i", str(idx), "-rgc"])
    refs = [r for r in runs if r["is_reference"]]
    per = {}
    for c in conds:
        occ = [r for r in runs if r["condition"] == c]
        rec = {"occurrences": len(occ)}
        for k in METRICS:
            allreps = [s for r in occ for s in r[k]["samples"]]
            ratios = []
            for r in occ:
                if c == "baseline":
                    continue
                dmin = min(abs(x["pos"] - r["pos"]) for x in refs)
                near = [x for x in refs if abs(x["pos"] - r["pos"]) == dmin]
                ref_med = statistics.mean(x[k]["median"] for x in near)
                ratios.append(r[k]["median"] / ref_med)
            rec[k] = {"median_all_reps_ms": statistics.median(allreps), "n_reps": len(allreps),
                      "occurrence_medians_ms": [r[k]["median"] for r in occ],
                      "drift_adjusted_ratio": math.exp(statistics.mean(math.log(x) for x in ratios)) if ratios else 1.0,
                      "occurrence_ratios": ratios,
                      "peak_alloc_bytes": max(r[k]["max_memory_allocated_bytes"] for r in occ),
                      "peak_reserved_bytes": max(r[k]["max_memory_reserved_bytes"] for r in occ)}
        per[c] = rec
    clocks = [g for r in runs for k in METRICS for g in r[k]["per_rep_gpu"]]
    payload = {"sequence": seq, "per_condition": per, "runs": runs, "clock_lock": lock, "sampler": sampler_kind,
               "clock_summary": {f: {"min": min(g[f] for g in clocks), "max": max(g[f] for g in clocks),
                                     "median": statistics.median(g[f] for g in clocks)}
                                 for f in ("sm_mhz", "mem_mhz", "temp_c", "power_w")},
               "reference_baseline_medians": {k: [x[k]["median"] for x in refs] for k in METRICS},
               "drift_rule": "ratio of each occurrence median to the mean of the nearest baseline reference(s) by "
                             "sequence position; geometric mean over the condition's occurrences",
               "seconds": time.time() - t0, "exclusive": True,
               "execution": "dense FP16 kernels for every condition (zeros / fake-quant weights)"}
    meta = collect_meta(seed=6304, dtype=torch.float16, models=["base"],
                        extra={"timing_exclusive": True, "other_gpu_state": resources.gpu_state()})
    save_result(ROOT / "results/task1" / a.out, payload, meta)
    for c in conds:
        p = ROOT / "results/task1" / c / "summary.json"
        if a.conditions and c == "baseline":
            continue  # short pass: the baseline summary keeps the full pass; refs are in the short-pass file
        if not p.exists():
            continue
        m0, s = load_result(p)
        if "latency_memory" in s:
            s["latency_memory_superseded"] = {**s.pop("latency_memory"),
                                              "superseded_by": "results/task1/retiming.json",
                                              "reason": "order-dependent clock/thermal drift; see docs/deviations.md"}
        s["latency_retimed"] = {k: per[c][k] for k in METRICS}
        s["latency_retimed_source"] = f"results/task1/{a.out}"
        save_result(p, s, m0)
    print("clock lock:", lock, flush=True)


if __name__ == "__main__":
    main()
