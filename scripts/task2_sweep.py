#!/usr/bin/env python3
"""Task 2 Part B exclusive measurements (other GPU idle):
1) forward-only inference sweep of each attention operator in isolation (one Attention module: qkv projection +
   operator + output projection) on fixed random embeddings, T in {256,1024,4096,8192}, B=4, H=4, d=32, FP32,
   >= 10 warmed reps; records which implementations materialise dense masks or linear prefix tensors, and OOMs.
2) clean training-step time per architecture (B=32, T=256, 20 warmup + 50 timed steps, alone on the GPU).
Writes results/task2/partB/inference_sweep.jsonl and train_step_timing.json."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta  # noqa: E402
from atml4.common.timing import read_peak_memory, reset_peak_memory, time_cuda  # noqa: E402
from atml4.task1.resources import gpu_state  # noqa: E402
from atml4.task2 import mqar_model as Mm  # noqa: E402

TS = [256, 1024, 4096, 8192]
B = 4
OPS = {"dense_sdpa_causal": ("dense", False, "none (is_causal=True, no mask tensor)"),
       "window_dense_mask": ("window", False, "EXPLICIT dense boolean mask [H,T,T]"),
       "dilated_mixed_dense_mask": ("dilated", False, "EXPLICIT dense boolean mask [H,T,T]"),
       "linear_cumsum": ("linear", False, "MATERIALISES prefix tensors S_t: [B,H,T,32,32]"),
       "linear_recurrent": ("linear", True, "O(1) state S [B,H,32,32], z [B,H,32]; Python loop over T")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.cuda.set_device(dev)
    rows = []
    for T in TS:
        g = torch.Generator(device=dev).manual_seed(6304 + T)
        x = torch.randn(B, T, Mm.D_MODEL, device=dev, generator=g)
        for name, (arch, rec, mat) in OPS.items():
            torch.manual_seed(6304)
            att = Mm.Attention(arch).to(dev).eval()
            att.recurrent = rec
            row = {"op": name, "T": T, "B": B, "H": Mm.N_HEADS, "d": Mm.D_HEAD, "dtype": "fp32",
                   "materialisation": mat, "unit": "ms"}
            if "dense mask" in mat:
                row["mask_bytes"] = Mm.N_HEADS * T * T  # torch.bool = 1 byte
            if name == "linear_cumsum":
                row["prefix_tensor_bytes"] = B * Mm.N_HEADS * T * Mm.D_HEAD * Mm.D_HEAD * 4
            try:
                reset_peak_memory(dev)
                with torch.no_grad():
                    r = time_cuda(lambda: att(x), reps=10, warmup=3, min_reps=10, device=dev)
                row.update({"status": "ok", **r.to_dict(), **read_peak_memory(dev)})
            except Exception as e:
                oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
                row.update({"status": "OOM" if oom else "error", "error": str(e)[:300]})
            torch.cuda.empty_cache()
            rows.append(row)
            print(T, name, row["status"], row.get("median"), flush=True)
    meta = collect_meta(seed=6304, extra={"timing_exclusive": True, "other_gpu_state": gpu_state()})
    save_result(ROOT / "results/task2/partB/inference_sweep.jsonl", rows, meta)

    from atml4.task2.mqar_model import MQARModel, lr_at, query_loss
    steps = {}
    g = torch.Generator().manual_seed(6304)
    tok = torch.randint(2, 192, (32, 256), generator=g).to(dev)
    qp = torch.stack([torch.randperm(200, generator=g)[:16] + 40 for _ in range(32)]).to(dev)
    ans = torch.randint(128, 192, (32, 16), generator=g).to(dev)
    for arch in Mm.ARCHS:
        torch.manual_seed(6304)
        m = MQARModel(arch).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=lr_at(0), weight_decay=0.01)

        def step():
            loss = query_loss(m(tok, qp), ans)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        reset_peak_memory(dev)
        r = time_cuda(step, reps=50, warmup=20, min_reps=10, device=dev)
        steps[arch] = {**r.to_dict(), **read_peak_memory(dev), "batch": 32, "T": 256, "K": 16}
        print(arch, "train step median ms", r.median, flush=True)
    save_result(ROOT / "results/task2/partB/train_step_timing.json",
                {"per_arch": steps, "theory": Mm.theoretical_access(256), "state_bytes_T256_fp32": Mm.state_bytes(256),
                 "state_bytes_T256_fp16": Mm.state_bytes(256, 2)}, meta)


if __name__ == "__main__":
    main()
