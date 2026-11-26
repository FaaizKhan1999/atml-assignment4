"""Pinned Qwen2.5-1.5B-Instruct loader for Task 2C (FP16, SDPA)."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atml4.common.meta import model_revision


def load(device):
    mid, rev = model_revision("target")
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=torch.float16, attn_implementation="sdpa")
    model.to(device).eval().requires_grad_(False)
    return model, tok
