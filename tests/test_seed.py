import random

import numpy as np
import pytest

from atml4.common.seed import DEFAULT_SEED, set_seed


def _draw():
    out = [random.random(), float(np.random.rand())]
    try:
        import torch
        out.append(float(torch.rand(1)))
    except ImportError:
        pass
    return out


def test_default_seed():
    assert DEFAULT_SEED == 6304


def test_reproducible():
    set_seed(6304)
    a = _draw()
    set_seed(6304)
    b = _draw()
    set_seed(6305)
    c = _draw()
    assert a == b
    assert a != c


def test_deterministic_flags():
    torch = pytest.importorskip("torch")
    set_seed(6304, deterministic=True)
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
