"""Shared latency/memory measurement for Task 1 (prefill 512/2048, 64-token greedy decode after 512 prefill)."""
from __future__ import annotations

import json
import subprocess

import torch

from atml4.common.timing import read_peak_memory, reset_peak_memory, time_cuda

TIMING_SEED = 6304
TIMING_LEN = 2048


def make_timing_inputs(tok, n: int = TIMING_LEN, seed: int = TIMING_SEED) -> dict:
    """Uniform random token IDs from the regular vocabulary: ids in [0, tok.vocab_size) minus all special IDs
    (added tokens >= tok.vocab_size are excluded too). The 512-token prompt is the first 512 of the 2048."""
    special = set(tok.all_special_ids)
    cand = torch.tensor([i for i in range(tok.vocab_size) if i not in special], dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    ids = cand[torch.randint(len(cand), (n,), generator=g)].tolist()
    return {"seed": seed, "n": n, "vocab_rule": "[0, tokenizer.vocab_size) minus all_special_ids",
            "tokenizer_vocab_size": tok.vocab_size, "n_candidates": len(cand), "token_ids": ids}


def gpu_state() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"unavailable: {e}"


@torch.no_grad()
def measure(model, token_ids: list[int], device, reps: int = 5, warmup: int = 2, decode_steps: int = 64) -> dict:
    dev = torch.device(device)
    torch.cuda.set_device(dev)
    ids = torch.tensor(token_ids, dtype=torch.long, device=dev)[None]
    out = {"unit": "ms", "reps": reps, "warmup": warmup, "batch": 1, "gpu_state_before": gpu_state()}
    for L in (512, 2048):
        x = ids[:, :L]
        reset_peak_memory(dev)
        r = time_cuda(lambda: model(x, use_cache=True), reps=reps, warmup=warmup, device=dev)
        out[f"prefill_{L}"] = {**r.to_dict(), **read_peak_memory(dev)}

    x = ids[:, :512]

    def setup():
        o = model(x, use_cache=True)
        return o.past_key_values, o.logits[:, -1:].argmax(-1)

    def decode(state):
        past, nxt = state
        for _ in range(decode_steps):  # greedy, EOS ignored
            o = model(nxt, past_key_values=past, use_cache=True)
            past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        return nxt

    reset_peak_memory(dev)
    r = time_cuda(decode, setup=setup, reps=reps, warmup=warmup, device=dev)
    d = r.to_dict()
    d.update({"prefill_tokens_untimed": 512, "decode_steps": decode_steps,
              "ms_per_token_mean": r.mean / decode_steps, "ms_per_token_median": r.median / decode_steps,
              **read_peak_memory(dev)})
    out["decode_64_after_512"] = d
    out["gpu_state_after"] = gpu_state()
    return out


def sdpa_backend_in_model(model, device, L: int = 512) -> dict:
    """Profile one forward and report which SDPA kernel family actually ran."""
    from torch.profiler import ProfilerActivity, profile

    x = torch.zeros(1, L, dtype=torch.long, device=device)
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        model(x, use_cache=False)
        torch.cuda.synchronize()
    names = sorted({e.key for e in prof.key_averages()})
    attn = [n for n in names if any(s in n.lower() for s in ("fmha", "flash", "attention", "sdpa", "cudnn"))]
    j = " ".join(attn).lower()
    backend = ("flash" if "flash" in j else "efficient" if ("fmha" in j or "efficient" in j) else
               "cudnn" if "cudnn" in j else "math/other")
    return {"backend": backend, "attention_kernels": attn, "L": L,
            "attn_implementation": getattr(model.config, "_attn_implementation", None)}


def load_or_create_timing_inputs(path, tok) -> dict:
    from pathlib import Path

    p = Path(path)
    fresh = make_timing_inputs(tok)
    if p.exists():
        saved = json.loads(p.read_text())
        saved = saved.get("payload", saved)
        if saved["token_ids"] != fresh["token_ids"]:
            raise RuntimeError("saved timing inputs differ from regeneration; refusing to continue")
        return saved
    return fresh
