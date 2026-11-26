#!/usr/bin/env python3
"""Kaggle T4x2 orchestration for the Task 2C full run: quality shards on cuda:0 || cuda:1 (non-exclusive), then the
exclusive timing subset and the packed-cache benchmark on cuda:0 with cuda:1 idle."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def log(m):
    print(f"[t2c {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = {"commit": commit, "start": time.time(), "steps": []}
    (ROOT / "results/task2/partC").mkdir(parents=True, exist_ok=True)
    t = time.time()
    procs = []
    for i in (0, 1):
        lf = open(ROOT / f"results/task2/partC/quality_s{i}.log", "w")
        procs.append(subprocess.Popen([PY, "scripts/task2c_run.py", "quality", "--device", f"cuda:{i}", "--shard",
                                       str(i), "--nshards", "2"], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    status["steps"].append({"name": "quality_parallel", "rcs": rcs, "seconds": time.time() - t})
    log(f"quality rcs={rcs} {time.time() - t:.0f}s")
    for mode in ("timing", "bench"):
        t = time.time()
        rc = subprocess.call([PY, "scripts/task2c_run.py", mode, "--device", "cuda:0"], cwd=ROOT)
        status["steps"].append({"name": f"{mode}_exclusive_cuda0", "rc": rc, "seconds": time.time() - t})
        log(f"{mode} rc={rc} {time.time() - t:.0f}s")
    status["total_seconds"] = time.time() - status["start"]
    (ROOT / "results/task2/partC/run_status.json").write_text(json.dumps(status, indent=1))
    sys.exit(1 if any(s.get("rc") for s in status["steps"]) or any(rcs) else 0)


if __name__ == "__main__":
    main()
