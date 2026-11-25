#!/usr/bin/env python3
"""Tiny probe kernel: report what accelerator Kaggle actually provided, then hold the session for
--hold seconds so that concurrency of two kernels can be observed with `kaggle kernels status`."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--hold", type=int, default=240)
a = ap.parse_args()
t0 = time.time()
smi = subprocess.run("nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv",
                     shell=True, capture_output=True, text=True).stdout
print(smi, flush=True)
time.sleep(a.hold)
save_result(ROOT / f"results/env/accel_probe_{a.tag}.json",
            {"nvidia_smi": smi, "start_epoch": t0, "end_epoch": time.time(), "hold_s": a.hold},
            collect_meta())
