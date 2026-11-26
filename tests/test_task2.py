"""Task 2 Part A/B CPU tests."""
import pytest

torch = pytest.importorskip("torch")

from atml4.task2 import attention as A  # noqa: E402
from atml4.task2 import mqar_data, mqar_model as M  # noqa: E402


def _qkv(L, d=16, B=2, H=3, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, L, d, generator=g, dtype=dtype) for _ in range(4)]


@pytest.mark.parametrize("L,tile", [(64, 16), (100, 32), (37, 64), (128, 128)])
def test_tiled_equals_dense_fp32(L, tile):
    q, k, v, go = _qkv(L)
    ref = A.dense_attention(q, k, v)
    assert torch.allclose(A.tiled_attention(q, k, v, tile), ref, atol=1e-5)
    assert torch.allclose(A.sdpa(q, k, v), ref, atol=1e-5)
    r_out, r_g = A.dense_reference_fp32(q, k, v, go, chunk=24)
    assert torch.allclose(r_out, ref, atol=1e-5)
    qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
    A.tiled_attention(qq, kk, vv, tile).backward(go)
    for g1, g2 in zip((qq.grad, kk.grad, vv.grad), r_g):
        assert torch.allclose(g1, g2, atol=1e-4)


@pytest.mark.parametrize("op", ["dense", "tiled", "sdpa"])
def test_causality_bit_identical(op):
    q, k, v, _ = _qkv(64)
    fn = {"dense": A.dense_attention, "tiled": lambda a, b, c: A.tiled_attention(a, b, c, 16), "sdpa": A.sdpa}[op]
    k2, v2 = k.clone(), v.clone()
    k2[..., 40:, :] += 3.0
    v2[..., 40:, :] -= 2.0
    a, b = fn(q, k, v), fn(q, k2, v2)
    assert torch.equal(a[..., :40, :], b[..., :40, :])
    assert not torch.equal(a[..., 40:, :], b[..., 40:, :])


def test_all_masked_rows_zero_and_zero_grad():
    q, k, v, _ = (t.requires_grad_(True) for t in _qkv(8, B=1, H=1))
    kpm = torch.ones(1, 8, dtype=torch.bool)
    kpm[:, :3] = False
    o = A.dense_attention(q, k, v, key_padding_mask=kpm)
    o.sum().backward()
    assert (o[..., :3, :] == 0).all() and torch.isfinite(o).all()
    assert (q.grad[..., :3, :] == 0).all() and (k.grad[..., :3, :] == 0).all() and (v.grad[..., :3, :] == 0).all()
    assert torch.isfinite(q.grad).all() and (o[..., 3:, :] != 0).any()


def test_naive_softmax_overflows_stable_does_not():
    q, k, v, _ = _qkv(16, B=1, H=1)
    q, k = q * 40, k * 40
    assert torch.isfinite(A.dense_attention(q, k, v, stable=True)).all()
    assert not torch.isfinite(A.dense_attention(q, k, v, stable=False)).all()


def test_linear_recurrence_equals_cumsum():
    q, k, v, _ = _qkv(50, d=8)
    assert torch.allclose(M.linear_attention_cumsum(q, k, v), M.linear_attention_recurrent(q, k, v), atol=1e-5)


def test_masks_and_receptive_field():
    T = 300
    w = M.window_mask(T)
    i = 200
    assert w[i].nonzero().flatten().tolist() == list(range(i - 63, i + 1))
    assert w[10].nonzero().flatten().tolist() == list(range(0, 11))
    d = M.dilated_mask(T)
    assert d[i].nonzero().flatten().tolist() == sorted(i - o for o in range(0, 253, 4) if i - o >= 0)
    assert d[290].sum() == 64 and (290 - d[290].nonzero().flatten()).max() == 252
    assert not torch.triu(w, 1).any() and not torch.triu(d, 1).any()
    hm = M.head_masks("dilated", T)
    assert torch.equal(hm[0], w) and torch.equal(hm[3], d)
    two = (w.float() @ w.float()) > 0
    reach = [(i - two[i].nonzero().min()).item() for i in range(T)]
    assert max(reach) == 2 * (M.WINDOW - 1) == 126
    th = M.theoretical_access(256)
    assert th["pairs_total_per_head"]["dense"] == 256 * 257 // 2
    assert th["pairs_total_per_head"]["window"] == sum(min(i + 1, 64) for i in range(256))


def test_lr_schedule():
    assert M.lr_at(0) == pytest.approx(8e-4 / 200)
    assert M.lr_at(199) == pytest.approx(8e-4)
    assert M.lr_at(200) == pytest.approx(8e-4)
    assert M.lr_at(4500) == pytest.approx(0.15 * 8e-4)
    assert M.lr_at(2350) == pytest.approx(8e-4 * (1 - 0.85 * 0.5))


@pytest.mark.parametrize("arch", M.ARCHS)
def test_loss_only_at_query_positions(arch):
    torch.manual_seed(0)
    m = M.MQARModel(arch).eval()
    g = torch.Generator().manual_seed(1)
    tok = torch.randint(2, 192, (2, 64), generator=g)
    qp = torch.tensor([[10, 20, 30], [5, 15, 25]])
    ans = torch.randint(128, 192, (2, 3), generator=g)
    logits = m(tok, qp)
    assert logits.shape == (2, 3, 64)
    l1 = M.query_loss(logits, ans)
    manual = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), (ans - 128).reshape(-1))
    assert float(l1) == pytest.approx(float(manual))
    tok2 = tok.clone()
    tok2[:, 31:] = 7  # tokens after the last query cannot affect any supervised logit (causal)
    assert torch.allclose(M.query_loss(m(tok2, qp), ans), l1, atol=1e-6)
    m.zero_grad()
    M.query_loss(m(tok, qp), ans).backward()
    assert torch.isfinite(m.emb.weight.grad).all()


def test_params_identical_across_archs():
    counts = {a: M.n_params(M.MQARModel(a)) for a in M.ARCHS}
    assert len(set(counts.values())) == 1


def test_generator_invariants_and_stream_rule():
    g = mqar_data._gen()
    ex = g.make_example("train", 16, "mixed", mqar_data.stream_index(6305, 3, 7))
    g.validate(ex)
    assert mqar_data.stream_index(6304, 4499, 31) < mqar_data.stream_index(6305, 0, 0)
    arr = mqar_data.generate_stream(4, "near", 6304, n_updates=2)
    assert arr["tokens"].shape == (64, 256) and arr["answer_ids"].shape == (64, 4)
    again = mqar_data.generate_stream(4, "near", 6304, n_updates=2)
    assert (arr["tokens"] == again["tokens"]).all()
    rows = mqar_data.load_split("development")
    g.validate(rows[0])
    assert len(rows) == 2304


def test_curriculum_schedule_and_examples():
    import importlib.util

    import numpy as np

    from atml4.task2 import mqar_data

    spec = importlib.util.spec_from_file_location("cur", mqar_data.repo_root() / "scripts/task2_curriculum.py")
    cur = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cur)
    assert cur.p_mixed(0) == 0 and cur.p_mixed(1499) == 0 and cur.p_mixed(1500) == 0
    assert cur.p_mixed(2250) == 0.5 and cur.p_mixed(3000) == 1 and cur.p_mixed(4499) == 1
    m = cur.regime_mask(6305)
    assert m.shape == (4500, 32) and not m[:1500].any() and m[3000:].all()
    frac = m[1500:3000].mean()
    assert 0.45 < frac < 0.55
    assert np.array_equal(m, cur.regime_mask(6305)) and not np.array_equal(m, cur.regime_mask(6304))
    g = mqar_data._gen()
    for u, i in ((10, 3), (2000, 7), (4000, 31)):
        reg = "mixed" if m[u, i] else "near"
        ex = g.make_example("train", 16, reg, mqar_data.stream_index(6305, u, i))
        assert ex["regime"] == reg and ex["K"] == 16
