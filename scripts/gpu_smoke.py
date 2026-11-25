#!/usr/bin/env python3
"""GPU smoke test (Kaggle): env lock, meta, Qwen2.5-0.5B-Instruct FP16 forward, SDPA backend probe,
prefix-bank smoke timing (--limit-prefix-prompts 2 and 4) with extrapolation to the full 120-prompt bank.

Writes results/env/gpu_smoke.json and env/requirements-lock.txt.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from atml4.common.io import save_result  # noqa: E402
from atml4.common.meta import collect_meta, model_revision  # noqa: E402
from atml4.common.seed import set_seed  # noqa: E402
from atml4.common.timing import WallTimer, read_peak_memory, reset_peak_memory, time_cuda  # noqa: E402

DATA_REPO = ROOT / "external/atml-pa4-datasets"
N_FULL_PROMPTS = 120  # 60 GSM8K test + 60 short-form test (setup_data.prefix_banks)


def freeze_env() -> str:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    header = ("# Frozen with `pip freeze` inside the Kaggle GPU smoke kernel (scripts/gpu_smoke.py).\n"
              "# No course environment lock was provided; this is the Kaggle runtime lock.\n")
    (ROOT / "env").mkdir(exist_ok=True)
    (ROOT / "env/requirements-lock.txt").write_text(header + out)
    return out


def forward_pass() -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid, rev = model_revision("draft")
    with WallTimer() as t_load:
        tok = AutoTokenizer.from_pretrained(mid, revision=rev)
        model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, torch_dtype=torch.float16,
                                                     attn_implementation="sdpa").to("cuda").eval()
    text = tok.apply_chat_template([{"role": "user", "content": "What is 2+3?"}], tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    reset_peak_memory()
    with torch.inference_mode():
        logits = model(ids).logits
    out = {"model": mid, "revision": rev, "load_ms": t_load.ms, "input_len": int(ids.shape[1]),
           "logits_shape": list(logits.shape), "logits_dtype": str(logits.dtype),
           "logits_finite": bool(torch.isfinite(logits).all()),
           "argmax_next_token": tok.decode(int(logits[0, -1].argmax())),
           "attn_implementation": getattr(model.config, "_attn_implementation", None),
           "chat_template_text": text, **read_peak_memory()}
    del model
    torch.cuda.empty_cache()
    return out


def sdpa_probe(L: int = 1024) -> dict:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    g = torch.Generator(device="cuda").manual_seed(6304)
    B, H, d = 2, 4, 64
    q, k, v = (torch.randn(B, H, L, d, device="cuda", dtype=torch.float16, generator=g) for _ in range(3))
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    res = {"shape": [B, H, L, d], "dtype": "float16", "is_causal": True, "forced": {}}
    for be in [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION,
               SDPBackend.MATH]:
        try:
            with sdpa_kernel(be):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                tr = time_cuda(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), reps=10, warmup=3)
            res["forced"][be.name] = {"ok": True, "max_abs_err_vs_fp32": float((o.float() - ref).abs().max()),
                                      "fwd_ms": tr.to_dict()}
        except Exception as e:
            res["forced"][be.name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}
    # Which backend the default dispatcher picks
    try:
        choice = torch._fused_sdp_choice(q, k, v, is_causal=True)
        res["default_fused_sdp_choice"] = SDPBackend(choice).name
    except Exception as e:
        res["default_fused_sdp_choice"] = f"unavailable: {e}"
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
    names = sorted({e.key for e in prof.key_averages()})
    res["default_profiled_kernels"] = names
    joined = " ".join(names).lower()
    res["default_backend_from_profile"] = ("flash" if "flash" in joined else
                                           "efficient" if ("fmha" in joined or "efficient" in joined or "mem_eff" in joined) else
                                           "cudnn" if "cudnn" in joined else "math/other")
    return res


def prefix_bank_smoke(limits=(2, 4)) -> dict:
    runs = []
    for n in limits:
        cmd = [sys.executable, "scripts/setup_data.py", "--prefix-banks", "--force", "--limit-prefix-prompts", str(n)]
        t0 = time.time()
        lines = []
        p = subprocess.Popen(cmd, cwd=DATA_REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            lines.append((round(time.time() - t0, 2), line.rstrip()))
        rc = p.wait()
        runs.append({"limit": n, "rc": rc, "seconds": round(time.time() - t0, 2), "timestamped_log": lines[-60:]})
    # Smoke banks must never be mistaken for the evaluation bank.
    smoke = DATA_REPO / "data/prefix_banks/prefix_banks_locked.jsonl.gz"
    if smoke.exists():
        smoke.unlink()
    est = {}
    ok = [r for r in runs if r["rc"] == 0]
    if len(ok) == 2:
        (n1, t1), (n2, t2) = [(r["limit"], r["seconds"]) for r in ok]
        per_prompt = (t2 - t1) / (n2 - n1)  # both models, one prompt
        fixed = t1 - n1 * per_prompt
        est = {"per_prompt_both_models_s": per_prompt, "fixed_overhead_s": fixed,
               "full_estimate_s": fixed + N_FULL_PROMPTS * per_prompt,
               "full_estimate_h": (fixed + N_FULL_PROMPTS * per_prompt) / 3600,
               "caveat": "smoke prompts are the first GSM8K test prompts (long math responses, 192-token cap); "
                         "the 60 short-form prompts likely stop earlier at EOS, so this is closer to an upper bound"}
    return {"runs": runs, "estimate": est}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-prefix", action="store_true")
    a = ap.parse_args()
    set_seed(6304, deterministic=False)
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    print(smi)
    freeze_env()
    meta = collect_meta(seed=6304, dtype=torch.float16, models=["draft"])
    payload = {"nvidia_smi": smi, "forward": forward_pass(), "sdpa": sdpa_probe()}
    lock = DATA_REPO / "data/local_source_revisions.json"
    if lock.exists():
        import json
        import yaml

        data_rev = json.loads(lock.read_text())["revisions"]
        ours = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["models"]
        payload["revision_consistency"] = {k: {"configs": ours[k2]["revision"], "data_lock": data_rev[k]}
                                           for k, k2 in [("model_base", "base"), ("model_target", "target"),
                                                         ("model_draft", "draft")]}
    if not a.skip_prefix:
        payload["prefix_bank_smoke"] = prefix_bank_smoke()
    save_result(ROOT / "results/env/gpu_smoke.json", payload, meta)
    print(re.sub(r"\s+", " ", str({k: payload["sdpa"].get(k) for k in ("default_fused_sdp_choice",
                                                                       "default_backend_from_profile")})))
    print("estimate:", payload.get("prefix_bank_smoke", {}).get("estimate"))


if __name__ == "__main__":
    main()
