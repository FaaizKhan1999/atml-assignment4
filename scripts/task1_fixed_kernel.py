#!/usr/bin/env python3
"""Kaggle T4x2 orchestration for Task 1 fixed conditions (P1, P2, 2:4, Q1, Q2).

0. lm-eval side check (cuda:0, alone)
1. parallel quality/compression: worker A (cuda:0) pruning family incl. pruning stats; worker B (cuda:1)
   quantization family. Phases storage,outerr,ppl,hellaswag (no timing).
2. barrier; then ALL timing sequentially on cuda:0 with cuda:1 idle (CLAUDE.md timing rule).
Status is written to results/task1/fixed_run_status.json.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atml4.task1.conditions import FIXED_CONDITIONS  # noqa: E402

PRUNE = [c for c in FIXED_CONDITIONS if c.startswith("p")]
QUANT = [c for c in FIXED_CONDITIONS if c.startswith("q")]
PY = sys.executable
QUALITY = "storage,outerr,ppl,hellaswag"


def log(msg):
    print(f"[orchestrator {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, name):
    t = time.time()
    log(f"start {name}: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=ROOT)
    log(f"end {name}: rc={rc} {time.time() - t:.0f}s")
    return {"name": name, "rc": rc, "seconds": time.time() - t, "cmd": cmd}


def main():
    status = {"steps": [], "start": time.time()}
    freeze = lambda: subprocess.run([PY, "-m", "pip", "freeze"], capture_output=True, text=True).stdout  # noqa
    before = freeze()
    status["steps"].append(run([PY, "-m", "pip", "install", "-q", "lm-eval==0.4.13"], "pip_lm_eval"))
    after = freeze()
    status["pip_changes_by_lm_eval"] = sorted(set(after.splitlines()) ^ set(before.splitlines()))
    status["steps"].append(run([PY, "scripts/task1_lmeval_sidecheck.py", "--device", "cuda:0", "--limit", "100"],
                               "lmeval_sidecheck"))

    t = time.time()
    la = open(ROOT / "results/task1/worker_A.log", "w")
    lb = open(ROOT / "results/task1/worker_B.log", "w")
    cmd_a = f"{PY} scripts/task1_pruning_stats.py --device cuda:0 && {PY} scripts/task1_eval.py --device cuda:0 " \
            f"--phases {QUALITY} --timing-exclusive no --conditions {' '.join(PRUNE)}"
    cmd_b = f"{PY} scripts/task1_eval.py --device cuda:1 --phases {QUALITY} --timing-exclusive no " \
            f"--conditions {' '.join(QUANT)}"
    log("parallel quality phase: A=" + cmd_a + " | B=" + cmd_b)
    pa = subprocess.Popen(cmd_a, shell=True, cwd=ROOT, stdout=la, stderr=subprocess.STDOUT)
    pb = subprocess.Popen(cmd_b, shell=True, cwd=ROOT, stdout=lb, stderr=subprocess.STDOUT)
    ra, rb = pa.wait(), pb.wait()
    la.close()
    lb.close()
    status["steps"].append({"name": "parallel_quality", "rc_A": ra, "rc_B": rb, "seconds": time.time() - t,
                            "cmd_A": cmd_a, "cmd_B": cmd_b})
    log(f"barrier reached: A rc={ra}, B rc={rb}, {time.time() - t:.0f}s")
    for f in ("worker_A.log", "worker_B.log"):
        print(f"----- tail {f}", flush=True)
        print("".join((ROOT / "results/task1" / f).read_text().splitlines(True)[-25:]), flush=True)

    status["steps"].append(run([PY, "scripts/task1_eval.py", "--device", "cuda:0", "--phases", "timing",
                                "--timing-exclusive", "yes", "--conditions", *FIXED_CONDITIONS],
                               "timing_exclusive_cuda0"))
    status["total_seconds"] = time.time() - status["start"]
    (ROOT / "results/task1/fixed_run_status.json").write_text(json.dumps(status, indent=1))
    bad = [s for s in status["steps"] if s.get("rc", 0) or s.get("rc_A", 0) or s.get("rc_B", 0)]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
