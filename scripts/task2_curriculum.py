#!/usr/bin/env python3
"""Task 2 B3 intervention (pre-registered, docs/decisions.md): reachable-query curriculum for window W=64, K=16.

  generate : per seed, regime mask r = default_rng(seed).random((4500, 32)) < p(u) with p = 0 (u < 1500),
             (u - 1500) / 1500 (1500 <= u < 3000), 1 (u >= 3000); example (u, i) = make_example('train', 16,
             'mixed' if mask else 'near', stream_index(seed, u, i)) -> element (u, i) of the B1 near / mixed K=16
             streams. Cached as <cache>/curriculum_K16_seed<seed>.npz; the mask is saved to
             results/task2/partB/curriculum/mask_s<seed>.npz (packbits).
  train    : the fixed Part B protocol (mqar_train.train_run, window arch) on that stream; evaluation on all 9 test
             cells exactly as task2_train.py; run name window_K16_curriculum_s<seed>.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from atml4.task2 import mqar_data  # noqa: E402
from atml4.task2.mqar_data import BATCH, UPDATES  # noqa: E402

SEEDS = (6304, 6305, 6306)
K = 16
OUT = ROOT / "results/task2/partB"


def p_mixed(u: int) -> float:
    if u < 1500:
        return 0.0
    if u < 3000:
        return (u - 1500) / 1500
    return 1.0


def regime_mask(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed).random((UPDATES, BATCH))
    p = np.array([p_mixed(u) for u in range(UPDATES)])[:, None]
    return r < p


def cache_path(cache: Path, seed: int) -> Path:
    return cache / f"curriculum_K{K}_seed{seed}.npz"


def gen(args):
    seed, cache = args
    g = mqar_data._gen()
    t = time.perf_counter()
    mask = regime_mask(seed)
    rows = [g.make_example("train", K, "mixed" if mask[u, i] else "near", mqar_data.stream_index(seed, u, i))
            for u in range(UPDATES) for i in range(BATCH)]
    arr = mqar_data.to_arrays(rows)
    arr["mixed_mask"] = mask
    arr["seconds"] = np.float64(time.perf_counter() - t)
    tmp = cache_path(cache, seed).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, cache_path(cache, seed))
    return seed, float(arr["seconds"])


def generate(a):
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    (OUT / "curriculum").mkdir(parents=True, exist_ok=True)
    if a.cpus:
        os.sched_setaffinity(0, a.cpus)
    with Pool(len(SEEDS)) as pool:
        for seed, s in pool.imap_unordered(gen, [(s, cache) for s in SEEDS]):
            print(f"curriculum stream seed {seed}: {s:.0f}s", flush=True)


def train(a):
    import torch

    from atml4.common.io import save_result
    from atml4.common.meta import collect_meta
    from atml4.task2 import mqar_train

    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    for sub in ("runs", "queries", "ckpt", "curriculum"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    dev_cells = mqar_data.cells(mqar_data.load_split("development"))
    test_cells = mqar_data.cells(mqar_data.load_split("test"))
    for seed in SEEDS:
        name = f"window_K{K}_curriculum_s{seed}"
        sp = cache_path(Path(a.cache), seed)
        while not sp.exists():
            time.sleep(10)
        t0 = time.time()
        stream = dict(np.load(sp))
        mask = stream.pop("mixed_mask")
        assert np.array_equal(mask, regime_mask(seed))
        np.savez_compressed(OUT / "curriculum" / f"mask_s{seed}.npz", packed=np.packbits(mask, axis=1),
                            shape=np.array(mask.shape))
        model, info = mqar_train.train_run("window", K, "mixed", seed, stream, dev_cells, dev,
                                           log=lambda m: print(f"[curriculum] {m}", flush=True))
        t_eval = time.time()
        test = mqar_train.evaluate(model, test_cells, dev, return_queries=True)
        mqar_train.write_queries(OUT / "queries" / f"{name}.jsonl.gz", test)
        info.update({"test": test, "test_overall_acc": mqar_train.overall(test), "eval_seconds": time.time() - t_eval,
                     "wall_seconds": time.time() - t0, "regime": "curriculum near->mixed",
                     "trained_on": {"K": K, "regime": "curriculum: near (u<1500), mix p(u)=(u-1500)/1500, mixed (u>=3000)"},
                     "dev_trained_cell": "K16_mixed (dev_trained_cell_acc in dev_curve)",
                     "comparator": f"window_K16_mixed_s{seed}",
                     "curriculum": {"mixed_examples_per_update": mask.sum(1).tolist(),
                                    "mixed_examples_total": int(mask.sum()), "near_examples_total": int((~mask).sum()),
                                    "mask_file": f"results/task2/partB/curriculum/mask_s{seed}.npz",
                                    "example_rule": "make_example('train', 16, 'mixed' if mask[u,i] else 'near', "
                                                    "(seed-6304)*10^7 + u*32 + i)",
                                    "stream_generation_seconds": float(stream["seconds"])},
                     "packing_note": "one run per GPU at a time; step times are not exclusive clean timings"})
        torch.save(model.state_dict(), OUT / "ckpt" / f"{name}.pt")
        save_result(OUT / "runs" / f"{name}.json", info, collect_meta(seed=seed, extra={"run": name, "device": a.device}))
        print(f"[curriculum] DONE {name} test={ {k: round(v['acc'], 4) for k, v in test.items() if k.startswith('K16')} }",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("generate", "train"))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cache", default="/tmp/mqar_curriculum")
    ap.add_argument("--cpus", type=int, nargs="*", default=None)
    a = ap.parse_args()
    (generate if a.mode == "generate" else train)(a)


if __name__ == "__main__":
    main()
