#!/usr/bin/env python3
"""Task 2 Part B worker: claims runs from the shared job list (atomic claim files), trains with the fixed PDF
protocol, evaluates the final checkpoint on all 9 frozen test cells (+ the W=64 test-time mask control for dense
mixed K=16), and writes results/task2/partB/runs/<run>.json, queries/<run>.jsonl.gz, ckpt/<run>.pt."""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.task2 import mqar_data, mqar_train  # noqa: E402

GROUPS = [("dense", 4, "mixed"), ("dense", 8, "mixed"), ("dense", 16, "mixed"), ("dense", 16, "near"),
          ("window", 4, "near"), ("window", 8, "near"), ("window", 16, "near"), ("window", 16, "mixed"),
          ("linear", 4, "mixed"), ("linear", 8, "mixed"), ("linear", 16, "mixed"),
          ("dilated", 4, "mixed"), ("dilated", 16, "mixed")]
SEEDS = (6304, 6305, 6306)
OUT = ROOT / "results/task2/partB"


def run_name(arch, K, regime, seed):
    return f"{arch}_K{K}_{regime}_s{seed}"


def all_jobs():
    jobs = [(a, K, r, s) for a, K, r in GROUPS for s in SEEDS]
    return sorted(jobs, key=lambda j: (0 if j[0] == "linear" else 1, -j[1]))  # heavier first


def claim(name) -> bool:
    d = OUT / "claims"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(d / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache", default="/tmp/mqar_streams")
    ap.add_argument("--worker", default="w0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    for sub in ("runs", "queries", "ckpt"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    dev_cells = mqar_data.cells(mqar_data.load_split("development"))
    test_cells = mqar_data.cells(mqar_data.load_split("test"))
    for arch, K, regime, seed in all_jobs():
        name = run_name(arch, K, regime, seed)
        if (OUT / "runs" / f"{name}.json").exists() or not claim(name):
            continue
        t0 = time.time()
        try:
            sp = mqar_data.stream_path(Path(a.cache), K, regime, seed)
            while not sp.exists():  # stream generation may still be running
                time.sleep(10)
            stream = dict(np.load(sp))
            model, info = mqar_train.train_run(arch, K, regime, seed, stream, dev_cells, dev,
                                               log=lambda m: print(f"[{a.worker}] {m}", flush=True))
            t_eval = time.time()
            test = mqar_train.evaluate(model, test_cells, dev, return_queries=True)
            mqar_train.write_queries(OUT / "queries" / f"{name}.jsonl.gz", test)
            info["test"] = test
            info["test_overall_acc"] = mqar_train.overall(test)
            if arch == "dense" and K == 16 and regime == "mixed":
                model.set_mask_override("window")
                info["test_with_window_mask_at_test_time"] = mqar_train.evaluate(model, test_cells, dev)
                model.set_mask_override(None)
            info["eval_seconds"] = time.time() - t_eval
            info["wall_seconds"] = time.time() - t0
            info["trained_on"] = {"K": K, "regime": regime}
            info["worker"] = a.worker
            info["packing_note"] = "trained with other runs sharing the GPU; step times are not clean timings"
            torch.save(model.state_dict(), OUT / "ckpt" / f"{name}.pt")
            save_result(OUT / "runs" / f"{name}.json", info,
                        collect_meta(seed=seed, extra={"run": name, "device": a.device}))
            print(f"[{a.worker}] DONE {name} test_overall={info['test_overall_acc']:.4f} "
                  f"wall={info['wall_seconds']:.0f}s", flush=True)
        except Exception as e:
            (OUT / "runs" / f"{name}.FAILED.json").write_text(json.dumps(
                {"run": name, "error": repr(e), "traceback": traceback.format_exc()}, indent=1))
            print(f"[{a.worker}] FAILED {name}: {e}", flush=True)


if __name__ == "__main__":
    main()
