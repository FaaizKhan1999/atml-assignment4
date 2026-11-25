"""Timing and peak-memory helpers.

Protocol (CLAUDE.md): warm up, synchronize around timed regions, >=5 reps (>=10 where specified), keep all raw
samples together with unit and rep count.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass
class TimingResult:
    samples: list[float]
    unit: str
    reps: int
    warmup: int
    device: str
    mean: float = field(init=False)
    median: float = field(init=False)
    std: float = field(init=False)
    min: float = field(init=False)
    max: float = field(init=False)

    def __post_init__(self):
        s = self.samples
        self.mean = statistics.fmean(s)
        self.median = statistics.median(s)
        self.std = statistics.stdev(s) if len(s) > 1 else 0.0
        self.min = min(s)
        self.max = max(s)

    def to_dict(self) -> dict:
        return asdict(self)


def _check(reps: int, min_reps: int):
    if reps < min_reps:
        raise ValueError(f"reps={reps} < required minimum {min_reps}")


def time_cuda(fn: Callable[[], object], *, reps: int = 5, warmup: int = 2, min_reps: int = 5,
              device=None) -> TimingResult:
    """Time fn() on CUDA with per-rep CUDA events; synchronizes before and after each timed region.

    Returns raw per-rep milliseconds.
    """
    import torch

    _check(reps, min_reps)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        samples.append(float(start.elapsed_time(end)))
    return TimingResult(samples=samples, unit="ms", reps=reps, warmup=warmup,
                        device=torch.cuda.get_device_name(device))


def time_cpu(fn: Callable[[], object], *, reps: int = 5, warmup: int = 2, min_reps: int = 5) -> TimingResult:
    """CPU fallback using time.perf_counter_ns. Returns raw per-rep milliseconds."""
    _check(reps, min_reps)
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return TimingResult(samples=samples, unit="ms", reps=reps, warmup=warmup, device="cpu")


def time_fn(fn, **kw) -> TimingResult:
    """CUDA timer if CUDA is available, else CPU fallback."""
    try:
        import torch

        if torch.cuda.is_available():
            return time_cuda(fn, **kw)
    except ImportError:
        pass
    kw.pop("device", None)
    return time_cpu(fn, **kw)


class WallTimer:
    """Synchronized wall-clock section timer (for calibration/training/prefill phases)."""

    def __init__(self, cuda_sync: bool = True):
        self.cuda_sync = cuda_sync
        self.ms: float | None = None

    def _sync(self):
        if self.cuda_sync:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except ImportError:
                pass

    def __enter__(self):
        self._sync()
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.ms = (time.perf_counter_ns() - self._t0) / 1e6
        return False


def reset_peak_memory(device=None) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def read_peak_memory(device=None) -> dict:
    """GPU-allocated bytes (allocator view), NOT logical/packed bytes."""
    import torch

    if not torch.cuda.is_available():
        return {"max_memory_allocated_bytes": None, "max_memory_reserved_bytes": None}
    torch.cuda.synchronize(device)
    return {"max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
