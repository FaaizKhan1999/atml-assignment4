#!/usr/bin/env python3
"""Kaggle T4x2 orchestration: Task 1 Q3-proj + Task 2 B3 curriculum (pre-registered in docs/decisions.md).

CPU (background): curriculum stream generation (3 processes).
Worker A (cuda:0): AWQ re-derivation b in {8,4} + assert -> probes blocks 0-14 -> wait for shard b -> exact MILP
                   + dev checks + primary choice -> evaluate q3proj_b4.
Worker B (cuda:1): AWQ re-derivation b in {3,2} + assert -> probes blocks 15-27 -> 3 curriculum runs + test
                   evaluation -> wait for the allocation -> evaluate q3proj_b3.
Barrier, then the exclusive ABBA short re-timing (baseline, q3proj_b4, q3proj_b3) on cuda:0 with cuda:1 idle.
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
    (ROOT / "results/task2/partB").mkdir(parents=True, exist_ok=True)
    glog = open(ROOT / "results/task2/partB/curriculum_streams.log", "w")
    pg = subprocess.Popen([PY, "scripts/task2_curriculum.py", "generate"], cwd=ROOT, stdout=glog,
                          stderr=subprocess.STDOUT)
    wait_alloc = (f"{PY} -c \"import time,pathlib;p=pathlib.Path('results/task1/q3proj_allocation.json');"
                  f"[time.sleep(20) for _ in iter(lambda: p.exists(), True)]\"")
    cmd_a = " && ".join([
        f"{PY} scripts/task1_q3proj.py probe --device cuda:0 --derive-bits 8 4 --blocks 0 14 --tag a",
        f"{PY} scripts/task1_q3proj.py solve --device cuda:0 --shards a b",
        f"{PY} scripts/task1_eval.py --device cuda:0 --phases {Q} --timing-exclusive no --conditions q3proj_b4",
    ])
    cmd_b = " && ".join([
        f"{PY} scripts/task1_q3proj.py probe --device cuda:1 --derive-bits 3 2 --blocks 15 27 --tag b",
        f"{PY} scripts/task2_curriculum.py train --device cuda:1",
        wait_alloc,
        f"{PY} scripts/task1_eval.py --device cuda:1 --phases {Q} --timing-exclusive no --conditions q3proj_b3",
    ])
    t = time.time()
    la = open(ROOT / "results/task1/q3proj_worker_A.log", "w")
    lb = open(ROOT / "results/task1/q3proj_worker_B.log", "w")
    log("A: " + cmd_a)
    log("B: " + cmd_b)
    pa = subprocess.Popen(cmd_a, shell=True, cwd=ROOT, stdout=la, stderr=subprocess.STDOUT)
    pb = subprocess.Popen(cmd_b, shell=True, cwd=ROOT, stdout=lb, stderr=subprocess.STDOUT)
    while pa.poll() is None or pb.poll() is None:
        time.sleep(20)
        # a failed probe shard would leave the other worker waiting forever
        for me, other, tag in ((pa, pb, "A"), (pb, pa, "B")):
            if me.poll() not in (None, 0) and other.poll() is None and \
                    not (ROOT / "results/task1/q3proj_allocation.json").exists():
                log(f"worker {tag} failed before the allocation existed; terminating the other worker")
                other.terminate()
    ra, rb = pa.returncode, pb.returncode
    rg = pg.wait()
    la.close()
    lb.close()
    glog.close()
    status["steps"].append({"name": "parallel", "rc_A": ra, "rc_B": rb, "rc_streams": rg, "seconds": time.time() - t,
                            "cmd_A": cmd_a, "cmd_B": cmd_b})
    log(f"barrier: A rc={ra} B rc={rb} streams rc={rg} after {time.time() - t:.0f}s")
    for f in ("results/task1/q3proj_worker_A.log", "results/task1/q3proj_worker_B.log"):
        print(f"----- tail {f}", flush=True)
        print("".join((ROOT / f).read_text().splitlines(True)[-25:]), flush=True)
    rc = None
    if all((ROOT / f"results/task1/{c}/summary.json").exists() for c in ("q3proj_b4", "q3proj_b3")):
        t = time.time()
        rc = subprocess.call([PY, "scripts/task1_retime.py", "--device", "cuda:0", "--conditions", "q3proj_b4",
                              "q3proj_b3", "--out", "retiming_q3proj.json"], cwd=ROOT)
        status["steps"].append({"name": "retime_short_exclusive_cuda0", "rc": rc, "seconds": time.time() - t})
    status["total_seconds"] = time.time() - status["start"]
    (ROOT / "results/task1/q3proj_run_status.json").write_text(json.dumps(status, indent=1))
    sys.exit(1 if (ra or rb or rc or rg) else 0)


if __name__ == "__main__":
    main()
