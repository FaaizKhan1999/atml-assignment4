"""Task 1 condition registry. Each condition: apply(model, ctx) -> info dict, run on freshly restored weights.

Later P1/P2/P3/Q1/Q2/Q3/2:4 conditions register here and MUST be evaluated by atml4.task1.runner unchanged.
"""
from __future__ import annotations

from typing import Callable

CONDITIONS: dict[str, Callable] = {}


def register(name: str):
    def deco(fn):
        if name in CONDITIONS:
            raise ValueError(f"duplicate condition {name}")
        CONDITIONS[name] = fn
        return fn
    return deco


@register("baseline")
def baseline(model, ctx) -> dict:
    """Unchanged FP16 weights."""
    return {"description": "unchanged FP16 Qwen2.5-1.5B base"}
