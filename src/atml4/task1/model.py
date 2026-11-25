"""Model loading, eligible-projection enumeration, pristine snapshot/restore for Task 1."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

ELIGIBLE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_base_model(device: str = "cuda:0", dtype=torch.float16):
    """Qwen/Qwen2.5-1.5B base (NOT Instruct) at the pinned revision, FP16, SDPA attention."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from atml4.common.meta import model_revision

    mid, rev = model_revision("base")
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(mid, revision=rev, dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    model.requires_grad_(False)
    return model, tok


@dataclass
class Proj:
    layer: int
    name: str
    qualname: str
    module: torch.nn.Linear
    shape: tuple  # (d_out, d_in)


def eligible_projections(model) -> list[Proj]:
    """All eligible Linear projections in every decoder block, in (layer, ELIGIBLE) order."""
    out = []
    for li, layer in enumerate(model.model.layers):
        for name in ELIGIBLE:
            parent = layer.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
            mod = getattr(parent, name)
            qual = f"model.layers.{li}.{'self_attn' if parent is layer.self_attn else 'mlp'}.{name}"
            out.append(Proj(li, name, qual, mod, tuple(mod.weight.shape)))
    return out


def eligible_weight_names(model) -> set[str]:
    return {p.qualname + ".weight" for p in eligible_projections(model)}


def tie_info(model) -> dict:
    emb = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    return {"config_tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
            "same_data_ptr": emb.data_ptr() == head.data_ptr(),
            "same_object": emb is head}


def unique_parameters(model) -> list[tuple[str, torch.Tensor]]:
    """Named parameters with shared/tied storage counted once (first name wins)."""
    seen, out = set(), []
    for n, p in model.named_parameters(remove_duplicate=False):
        key = (p.data_ptr(), p.numel(), p.dtype)
        if key in seen:
            continue
        seen.add(key)
        out.append((n, p))
    return out


def weights_sha256(model, names: list[str] | None = None) -> str:
    """sha256 over the raw bytes of eligible weights in enumeration order (exact restoration check)."""
    h = hashlib.sha256()
    for p in eligible_projections(model):
        w = p.module.weight.detach().contiguous().cpu()
        h.update(p.qualname.encode())
        h.update(w.view(torch.uint8).numpy().tobytes() if w.dtype != torch.uint8 else w.numpy().tobytes())
    return h.hexdigest()


class Pristine:
    """CPU copy of the unchanged eligible weights; restore() makes every condition start from it."""

    def __init__(self, model):
        self.model = model
        self.weights = {p.qualname: p.module.weight.detach().to("cpu", copy=True)
                        for p in eligible_projections(model)}
        self.sha256 = weights_sha256(model)

    def restore(self, verify: bool = True) -> None:
        with torch.no_grad():
            for p in eligible_projections(self.model):
                p.module.weight.copy_(self.weights[p.qualname].to(p.module.weight.device))
        if verify:
            self.verify()

    def verify(self) -> None:
        for p in eligible_projections(self.model):
            if not torch.equal(p.module.weight.detach().cpu(), self.weights[p.qualname]):
                raise RuntimeError(f"restoration mismatch at {p.qualname}")
