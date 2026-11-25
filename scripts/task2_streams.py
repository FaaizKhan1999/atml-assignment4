#!/usr/bin/env python3
"""Generate and cache the 18 MQAR training streams (6 (K, regime) x 3 seeds) with a process pool.
Also runs the course generator's invariant checks first and saves the output."""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task2 import mqar_data  # noqa: E402

STREAMS = [(K, r) for r in ("mixed", "near") for K in (4, 8, 16)]
SEEDS = (6304, 6305, 6306)


def work(args):
    K, regime, seed, cache, cpus = args
    if cpus:
        os.sched_setaffinity(0, cpus)
    p = mqar_data.stream_path(Path(cache), K, regime, seed)
    if p.exists():
        return (K, regime, seed, 0.0, "cached")
    t = time.perf_counter()
    mqar_data.generate_stream(K, regime, seed, out=p)
    return (K, regime, seed, time.perf_counter() - t, "generated")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/mqar_streams")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--exclude-cpu0", action="store_true", help="keep core 0 free (Part A runs pinned there)")
    a = ap.parse_args()
    cpus = sorted(set(range(os.cpu_count())) - {0}) if a.exclude_cpu0 and os.cpu_count() > 1 else None
    if cpus:
        os.sched_setaffinity(0, cpus)
    checks = mqar_data.invariant_checks()
    print("invariant checks:", checks, flush=True)
    t = time.time()
    jobs = [(K, r, s, a.cache, cpus) for s in SEEDS for K, r in STREAMS]
    jobs.sort(key=lambda j: -j[0] * (2 if j[1] == "mixed" else 1))  # heaviest first
    with Pool(a.procs) as pool:
        res = pool.map(work, jobs, chunksize=1)
    payload = {"invariant_checks": checks, "streams": [dict(zip(("K", "regime", "seed", "seconds", "status"), r))
                                                       for r in res],
               "wall_seconds": time.time() - t, "procs": a.procs, "cpu_affinity": cpus,
               "index_rule": "(seed - 6304) * 10**7 + update * 32 + i, split='train'"}
    save_result(ROOT / "results/task2/partB/streams.json", payload, collect_meta())
    print(f"streams ready in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
