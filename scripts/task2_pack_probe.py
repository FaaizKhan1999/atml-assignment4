#!/usr/bin/env python3
"""Measure whether packing n concurrent training processes on one GPU raises aggregate throughput.
Each process runs 150 steps of the heaviest arch (linear, K=16). Prints JSON with steps/s per n."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import sys,time,torch; sys.path.insert(0,'src')
from atml4.task2.mqar_model import MQARModel, query_loss
dev=torch.device(sys.argv[1]); torch.cuda.set_device(dev); torch.manual_seed(0)
arch=sys.argv[2]; m=MQARModel(arch).to(dev); o=torch.optim.AdamW(m.parameters(),lr=1e-4)
tok=torch.randint(2,192,(32,256),device=dev); qp=torch.randint(40,250,(32,16),device=dev); ans=torch.randint(128,192,(32,16),device=dev)
for i in range(170):
    if i==20: torch.cuda.synchronize(); t=time.time()
    l=query_loss(m(tok,qp),ans); o.zero_grad(); l.backward(); o.step()
torch.cuda.synchronize(); print(150/(time.time()-t))
"""


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    res = {}
    for arch in ("linear", "dense"):
        for n in (1, 2, 3, 4):
            t = time.time()
            ps = [subprocess.Popen([sys.executable, "-c", CHILD, dev, arch], cwd=ROOT, stdout=subprocess.PIPE,
                                   text=True) for _ in range(n)]
            rates = [float(p.communicate()[0].strip().splitlines()[-1]) for p in ps]
            res[f"{arch}_n{n}"] = {"per_proc_steps_s": rates, "aggregate_steps_s": sum(rates), "wall": time.time() - t}
    best = {}
    for arch in ("linear", "dense"):
        base = res[f"{arch}_n1"]["aggregate_steps_s"]
        best[arch] = max((1, 2, 3, 4), key=lambda n: res[f"{arch}_n{n}"]["aggregate_steps_s"]
                         if res[f"{arch}_n{n}"]["aggregate_steps_s"] >= 1.15 * base or n == 1 else 0)
    res["chosen_procs_per_gpu"] = min(best.values())
    print(json.dumps(res))
    (ROOT / "results/task2/partB").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/task2/partB/pack_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
