#!/usr/bin/env python3
"""Kaggle T4x2 orchestration completing Task 1 (P3, Q3, Q2 output-error fix, clean re-timing).

Worker A (cuda:0): P3 dev grid/selection -> P3@50 (HS 2000) + P3@75 (HS 512) -> Q2 output-error re-run ->
                   waits for Q3 allocation -> Q3@B3.
Worker B (cuda:1): AWQ search b in {8,4,3,2} + 112 dev probes + exact knapsack + dev checks -> Q3@B4.
Barrier, then the exclusive re-timing pass on cuda:0 with cuda:1 idle.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
Q = "storage,outerr,ppl,hellaswag"


def log(m):
    print(f"[orchestrator {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    log(f"running at commit {commit}")
    status = {"commit": commit, "start": time.time(), "steps": []}
    wait_alloc = (f"{PY} -c \"import time,pathlib;p=pathlib.Path('results/task1/q3_allocation.json');"
                  f"[time.sleep(20) for _ in iter(lambda: p.exists(), True)]\"")
    cmd_a = " && ".join([
        f"{PY} scripts/task1_p3.py --device cuda:0",
        f"{PY} scripts/task1_eval.py --device cuda:0 --phases {Q} --timing-exclusive no --conditions p3_owl_s50 p3_owl_s75",
        f"{PY} scripts/task1_eval.py --device cuda:0 --phases outerr --timing-exclusive no "
        f"--conditions q2_awq_int8 q2_awq_int4 q2_awq_int3",
        wait_alloc,
        f"{PY} scripts/task1_eval.py --device cuda:0 --phases {Q} --timing-exclusive no --conditions q3_mixed_b3",
    ])
    cmd_b = " && ".join([
        f"{PY} scripts/task1_q3.py --device cuda:1",
        f"{PY} scripts/task1_eval.py --device cuda:1 --phases {Q} --timing-exclusive no --conditions q3_mixed_b4",
    ])
    t = time.time()
    la, lb = open(ROOT / "results/task1/final_worker_A.log", "w"), open(ROOT / "results/task1/final_worker_B.log", "w")
    log("A: " + cmd_a)
    log("B: " + cmd_b)
    pa = subprocess.Popen(cmd_a, shell=True, cwd=ROOT, stdout=la, stderr=subprocess.STDOUT)
    pb = subprocess.Popen(cmd_b, shell=True, cwd=ROOT, stdout=lb, stderr=subprocess.STDOUT)
    rb = pb.wait()
    if rb != 0 and not (ROOT / "results/task1/q3_allocation.json").exists():
        log("worker B failed before writing the allocation; stopping worker A's wait")
        pa.terminate()
    ra = pa.wait()
    la.close()
    lb.close()
    status["steps"].append({"name": "parallel", "rc_A": ra, "rc_B": rb, "seconds": time.time() - t,
                            "cmd_A": cmd_a, "cmd_B": cmd_b})
    log(f"barrier: A rc={ra} B rc={rb} after {time.time() - t:.0f}s")
    for f in ("final_worker_A.log", "final_worker_B.log"):
        print(f"----- tail {f}", flush=True)
        print("".join((ROOT / "results/task1" / f).read_text().splitlines(True)[-30:]), flush=True)
    t = time.time()
    rc = subprocess.call([PY, "scripts/task1_retime.py", "--device", "cuda:0"], cwd=ROOT)
    status["steps"].append({"name": "retime_exclusive_cuda0", "rc": rc, "seconds": time.time() - t})
    status["total_seconds"] = time.time() - status["start"]
    (ROOT / "results/task1/final_run_status.json").write_text(json.dumps(status, indent=1))
    sys.exit(1 if (ra or rb or rc) else 0)


if __name__ == "__main__":
    main()
