#!/usr/bin/env python3
"""Kaggle T4x2 orchestration for Task 2 Parts A and B (A1-A3, B1, B2).

1. MQAR stream generation starts in the background (CPU cores 1..n-1) while
2. Part A runs EXCLUSIVELY on cuda:0 (pinned to CPU core 0; cuda:1 idle).
3. Wait for streams; packing probe on cuda:0 chooses processes per GPU (training only, not timing).
4. Training workers on cuda:0 and cuda:1 pull the 39 runs from a shared claim-file queue.
5. Exclusive inference sweep + clean train-step timing on cuda:0 (cuda:1 idle).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def log(m):
    print(f"[t2-orchestrator {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = {"commit": commit, "start": time.time(), "steps": []}
    log(f"commit {commit}")
    (ROOT / "results/task2/partB").mkdir(parents=True, exist_ok=True)
    slog = open(ROOT / "results/task2/partB/streams.log", "w")
    ps = subprocess.Popen([PY, "scripts/task2_streams.py", "--exclude-cpu0"], cwd=ROOT, stdout=slog,
                          stderr=subprocess.STDOUT)
    t = time.time()
    rc = subprocess.call(["taskset", "-c", "0", PY, "scripts/task2_partA.py", "--device", "cuda:0"], cwd=ROOT)
    status["steps"].append({"name": "partA_exclusive", "rc": rc, "seconds": time.time() - t,
                            "note": "stream generation ran concurrently on CPU cores 1..n-1"})
    log(f"Part A rc={rc} {time.time() - t:.0f}s; waiting for streams")
    rs = ps.wait()
    slog.close()
    status["steps"].append({"name": "streams", "rc": rs, "seconds": time.time() - status["start"]})
    log(f"streams rc={rs}")
    pp = subprocess.run([PY, "scripts/task2_pack_probe.py", "cuda:0"], cwd=ROOT, capture_output=True, text=True)
    try:
        n = json.loads(pp.stdout.strip().splitlines()[-1])["chosen_procs_per_gpu"]
    except Exception:
        n = 1
    status["steps"].append({"name": "pack_probe", "rc": pp.returncode, "procs_per_gpu": n})
    log(f"packing: {n} process(es) per GPU")
    t = time.time()
    procs = []
    for gi in (0, 1):
        for w in range(n):
            lf = open(ROOT / f"results/task2/partB/train_cuda{gi}_w{w}.log", "w")
            procs.append(subprocess.Popen([PY, "scripts/task2_train.py", "--device", f"cuda:{gi}",
                                           "--worker", f"g{gi}w{w}"], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    done = len(list((ROOT / "results/task2/partB/runs").glob("*_s630?.json")))
    failed = [p.name for p in (ROOT / "results/task2/partB/runs").glob("*.FAILED.json")]
    status["steps"].append({"name": "training", "rcs": rcs, "seconds": time.time() - t, "runs_done": done,
                            "runs_failed": failed})
    log(f"training done: {done} runs, failed {failed}, {time.time() - t:.0f}s")
    t = time.time()
    rc = subprocess.call([PY, "scripts/task2_sweep.py", "--device", "cuda:0"], cwd=ROOT)
    status["steps"].append({"name": "sweep_exclusive", "rc": rc, "seconds": time.time() - t})
    status["total_seconds"] = time.time() - status["start"]
    (ROOT / "results/task2/run_status.json").write_text(json.dumps(status, indent=1))
    log(f"all done in {status['total_seconds']:.0f}s")


if __name__ == "__main__":
    main()
