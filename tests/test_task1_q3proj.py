import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from atml4.task1 import awq, q3  # noqa: E402
from atml4.task1.model import Pristine, eligible_projections  # noqa: E402


def _instance(seed, U=6, shared=True):
    rng = random.Random(seed)
    bits = (2, 3, 4, 8)
    costs = [{b: rng.uniform(0, 0.2) / b + (rng.uniform(-1e-4, 1e-4) if b == 8 else 0) for b in bits}
             for _ in range(U)]
    weights = [{b: rng.randint(50, 80) * b for b in bits} for _ in range(U)]
    groups = []
    if shared:
        groups.append({"units": [0, 1, 2], "vec_of_bits": {2: "a", 3: "b", 4: "c", 8: "c"},
                       "vec_bytes": {"a": 30, "b": 30, "c": 30}})
        groups.append({"units": [3, 4], "vec_of_bits": {b: b for b in bits}, "vec_bytes": {b: 25 for b in bits}})
    lo = sum(min(w.values()) for w in weights)
    hi = sum(max(w.values()) for w in weights)
    return costs, weights, groups, rng.randint(lo + 80, (lo + hi) // 2 + 200)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("shared", [True, False])
def test_proj_milp_matches_brute_force(seed, shared):
    costs, weights, groups, budget = _instance(seed, shared=shared)
    bf = q3.brute_force(costs, weights, groups, budget, fixed_bytes=17)
    if bf is None:
        with pytest.raises(ValueError):
            q3.proj_milp(costs, weights, groups, budget, fixed_bytes=17)
        return
    sol = q3.proj_milp(costs, weights, groups, budget, fixed_bytes=17)
    assert sol["cost"] == pytest.approx(bf[0][0], abs=1e-9)
    assert sol["bytes"] == bf[0][1]  # tie-break: fewer bytes
    assert sol["bytes"] <= budget


def test_proj_milp_tie_prefers_fewer_bytes():
    costs = [{4: 0.0, 8: 0.0}, {4: 0.0, 8: 0.0}]
    weights = [{4: 10, 8: 20}, {4: 10, 8: 20}]
    sol = q3.proj_milp(costs, weights, [], 100)
    assert sol["bits"] == [4, 4] and sol["bytes"] == 20


def _res(seed, bits, distinct=True):
    g = torch.Generator().manual_seed(seed)
    res = {}
    for b in bits:
        res[b] = {"scales": {}}
        for li in range(2):
            for grp, d in (("qkv", 128), ("o", 128), ("gateup", 128), ("down", 256)):
                res[b]["scales"][f"L{li}.{grp}.b{b}"] = (torch.rand(d, generator=g) + 0.5).half().numpy()
    return res


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    m = Qwen2ForCausalLM(cfg).eval()
    m.requires_grad_(False)
    return m


def test_proj_apply_uniform_equals_block_apply(tiny):
    res = _res(1, (4, 2))
    pr = Pristine(tiny)
    awq.apply_quantized(tiny, None, [4, 2], res_by_bits=res)
    a = {p.qualname: p.module.weight.clone() for p in eligible_projections(tiny)}
    pr.restore()
    bm = {(li, n): (4 if li == 0 else 2) for li in range(2) for n in awq.PROJ_NAMES}
    info = awq.apply_quantized_proj(tiny, bm, res)
    for p in eligible_projections(tiny):
        assert torch.equal(p.module.weight, a[p.qualname]), p.qualname
    assert info["serialized_scale_bytes"] == 2 * 3 * 128 * 2
    pr.restore()
    pr.verify()


def test_proj_apply_single_projection_and_fold(tiny):
    res = _res(2, (4,))
    pr = Pristine(tiny)
    before = {p.qualname: p.module.weight.clone() for p in eligible_projections(tiny)}
    seen = []
    awq.apply_quantized_proj(tiny, {(1, "down_proj"): 4}, res, on_matrix=lambda q, qr: seen.append(q))
    assert seen == ["model.layers.1.mlp.down_proj"]
    for p in eligible_projections(tiny):
        changed = not torch.equal(p.module.weight, before[p.qualname])
        assert changed == (p.qualname in ("model.layers.1.mlp.down_proj", "model.layers.1.mlp.up_proj"))
    sd = torch.from_numpy(res[4]["scales"]["L1.down.b4"]).float()
    up = tiny.model.layers[1].mlp.up_proj.weight
    assert torch.equal(up, (before["model.layers.1.mlp.up_proj"].float() / sd[:, None]).half())
    eq = awq.unfolded_equivalent_proj({(1, "down_proj"): 4}, res)
    assert torch.allclose(eq("model.layers.1.mlp.up_proj", up), before["model.layers.1.mlp.up_proj"].float(),
                          rtol=2e-3, atol=1e-6)
    pr.restore()
    pr.verify()


def test_serialized_bytes_distinct_vectors():
    res = _res(3, (4, 8))
    bm = {(0, "q_proj"): 4, (0, "k_proj"): 8, (0, "v_proj"): 4, (0, "o_proj"): 8, (0, "gate_proj"): 4,
          (0, "up_proj"): 4, (0, "down_proj"): 8}
    assert awq.serialized_bytes_proj(bm, res, [0]) == (2 + 1 + 1) * 128 * 2
    res[8]["scales"]["L0.qkv.b8"] = res[4]["scales"]["L0.qkv.b4"].copy()  # identical vectors counted once
    assert awq.serialized_bytes_proj(bm, res, [0]) == (1 + 1 + 1) * 128 * 2
    assert np.array_equal(res[8]["scales"]["L0.qkv.b8"], res[4]["scales"]["L0.qkv.b4"])
