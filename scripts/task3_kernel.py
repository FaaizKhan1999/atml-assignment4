#!/usr/bin/env python3
"""Kaggle T4x2 orchestration for Task 3.

  a : phase 1 (non-timing, parallel): cuda:0 real-model checks -> speculative bank (iii) || cuda:1 teacher-response
      collection; phase 2 (EXCLUSIVE, cuda:1 idle): Step 2 baseline timing on cuda:0.
  b : (quality only) cuda:0 off_s6304 -> on_s6304 || cuda:1 off_s6305 -> on_s6305 (two runs per GPU, sequential);
      own-rollout banks (2 per GPU); then cuda:0 Step 5 alignment || cuda:1 standalone accuracy (5 models).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def log(m):
    print(f"[t3 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def par(cmds: dict, status, name):
    t = time.time()
    procs = {}
    for dev, seq in cmds.items():
        lf = open(ROOT / f"results/task3/{name}_{dev.replace(':', '')}.log", "w")
        procs[dev] = subprocess.Popen(" && ".join(seq), shell=True, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    rcs = {d: p.wait() for d, p in procs.items()}
    status["steps"].append({"name": name, "rcs": rcs, "seconds": time.time() - t, "cmds": cmds})
    log(f"{name} rcs={rcs} {time.time() - t:.0f}s")
    return all(v == 0 for v in rcs.values())


def main():
    which = sys.argv[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = {"commit": commit, "start": time.time(), "steps": []}
    (ROOT / "results/task3").mkdir(parents=True, exist_ok=True)
    a, b = "scripts/task3_a.py", "scripts/task3_b.py"
    ok = True
    if which == "a":
        ok = par({"cuda:0": [f"{PY} {a} checks --device cuda:0", f"{PY} {a} bank3 --device cuda:0"],
                  "cuda:1": [f"{PY} {a} collection --device cuda:1"]}, status, "a_phase1")
        if ok:
            ok = par({"cuda:0": [f"{PY} {a} baseline --device cuda:0"]}, status, "a_baseline_exclusive")
    else:
        ok = par({"cuda:0": [f"{PY} {b} distill --method off --seed 6304 --device cuda:0",
                             f"{PY} {b} distill --method on --seed 6304 --device cuda:0"],
                  "cuda:1": [f"{PY} {b} distill --method off --seed 6305 --device cuda:1",
                             f"{PY} {b} distill --method on --seed 6305 --device cuda:1"]}, status, "b_distill")
        if ok:
            ok = par({"cuda:0": [f"{PY} {b} ownbank --run off_s6304 --device cuda:0",
                                 f"{PY} {b} ownbank --run on_s6304 --device cuda:0"],
                      "cuda:1": [f"{PY} {b} ownbank --run off_s6305 --device cuda:1",
                                 f"{PY} {b} ownbank --run on_s6305 --device cuda:1"]}, status, "b_ownbank")
        if ok:
            ok = par({"cuda:0": [f"{PY} {b} align --device cuda:0"],
                      "cuda:1": [f"{PY} {b} accuracy --device cuda:1"]}, status, "b_align_accuracy")
    status["total_seconds"] = time.time() - status["start"]
    (ROOT / f"results/task3/run_status_{which}.json").write_text(json.dumps(status, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
