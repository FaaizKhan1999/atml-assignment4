"""Pruning masks, quantizer, packer and AWQ folding tests (CPU)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from atml4.task1 import awq, prune, quant, storage  # noqa: E402
from atml4.task1.model import Pristine, eligible_projections  # noqa: E402


def test_row_mask_counts_and_ties():
    g = torch.Generator().manual_seed(0)
    S = torch.rand(6, 16, generator=g)
    for s in (0.25, 0.5, 0.75):
        m = prune.row_mask(S, s)
        assert (m.sum(1) == int(s * 16)).all()
        # pruned entries are the lowest scores in each row
        assert (S[m].view(6, -1).max(1).values <= S[~m].view(6, -1).min(1).values).all()
    T = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.5, 2.0, 2.0, 2.0]])
    m = prune.row_mask(T, 0.25)  # k=2: 0.5 then the first 1.0 (lowest column index)
    assert m.tolist() == [[True, False, False, False, True, False, False, False]]
    with pytest.raises(AssertionError):
        prune.row_mask(torch.rand(2, 10), 0.25)
    for d_in in (1536, 8960):
        for s in (0.25, 0.5, 0.75):
            assert prune.k_per_row(s, d_in) == s * d_in


def test_two_four_structure_and_ties():
    g = torch.Generator().manual_seed(1)
    S = torch.rand(5, 32, generator=g)
    m = prune.two_four_mask(S)
    assert (m.view(5, 8, 4).sum(-1) == 2).all()
    assert int(m.sum()) == int(prune.row_mask(S, 0.5).sum()) == 5 * 16
    T = torch.tensor([[3.0, 1.0, 1.0, 1.0]])
    assert prune.two_four_mask(T).tolist() == [[False, True, True, False]]


def test_score_diagnostics_runs():
    g = torch.Generator().manual_seed(2)
    W = torch.randn(32, 64, generator=g)
    xn = torch.rand(64, generator=g) + 0.1
    d = prune.score_diagnostics(W, xn)
    assert -1 <= d["spearman_absW_wanda"] <= 1
    assert set(d["jaccard_pruned_mag_vs_wanda"]) == {"0.25", "0.5", "0.75"}
    assert 0 <= d["jaccard_pruned_mag_vs_wanda"]["0.5"] <= 1
    assert prune.score_diagnostics(W, torch.ones(64))["jaccard_pruned_mag_vs_wanda"]["0.5"] == 1.0


def test_quantizer_hand_example():
    # one group of 128: values 0..127 scaled; b=2 -> s=(127-0)/3, z=0
    w = torch.arange(128, dtype=torch.float32)[None, :] * 0.01
    qr = quant.quantize(w, 2)
    s = torch.tensor((1.27 - 0.0) / 3).half().float()
    assert qr.scale.item() == pytest.approx(s.item())
    assert qr.zero.item() == 0
    exp_codes = torch.clamp(torch.round(w / s), 0, 3)
    assert torch.equal(qr.codes.view(1, 128), exp_codes.long())
    assert torch.equal(qr.w_hat, (exp_codes * s).half())
    # negative-min group: z = round(-wmin/s)
    w2 = torch.linspace(-1.0, 0.5, 128)[None, :]
    q2 = quant.quantize(w2, 4)
    s2 = torch.tensor(1.5 / 15).half().float()
    z2 = torch.clamp(torch.round(torch.tensor(1.0) / s2), 0, 15)
    assert q2.zero.item() == z2.item()
    assert torch.equal(q2.codes.view(1, 128), torch.clamp(torch.round(w2 / s2) + z2, 0, 15).long())
    assert (q2.w_hat.float() - w2).abs().max() <= s2 / 2 + 1e-3
    # degenerate group
    q3 = quant.quantize(torch.full((1, 128), 0.25), 4)
    assert q3.degenerate_groups == 1 and q3.scale.item() > 0
    with pytest.raises(AssertionError):
        quant.quantize(torch.randn(2, 100), 4)
    qp = quant.quantize(torch.randn(2, 100), 4, allow_pad=True)
    assert qp.w_hat.shape == (2, 100) and qp.pad == 28


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_pack_roundtrip(bits):
    g = torch.Generator().manual_seed(bits)
    for shape in [(24, 256), (7, 384), (64, 128)]:
        W = (torch.randn(*shape, generator=g) * 0.05).half()
        qr = quant.quantize(W, bits)
        blob = quant.pack(qr, chunk_rows=8)
        assert len(blob) == quant.packed_nbytes(*shape, bits) == storage.quantized_matrix_bytes(
            *shape, bits=bits)["total_bytes"]
        c, z, s, w = quant.unpack(blob, *shape, bits, chunk_rows=8)
        assert torch.equal(c, qr.codes) and torch.equal(z, qr.zero)
        assert torch.equal(s.view(torch.int16), qr.scale.view(torch.int16))
        assert torch.equal(w.view(torch.int16), qr.w_hat.view(torch.int16))


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


def test_awq_folding_exact_unquantized(tiny):
    """Rescaled-but-unquantized model (fold down->up, runtime division elsewhere) reproduces outputs (fp32)."""
    pr = Pristine(tiny)
    fv = awq.check_fold_validity(tiny)
    assert fv["down_into_up_fold_valid"] and not fv["o_from_v_fold_valid"]  # GQA: v out 64 != o in 128
    g = torch.Generator().manual_seed(3)
    res = {"scales": {}}
    for li in range(2):
        for grp, d in (("qkv", 128), ("o", 128), ("gateup", 128), ("down", 256)):
            res["scales"][f"L{li}.{grp}.b4"] = (torch.rand(d, generator=g) * 3 + 0.2).numpy()
    x = torch.randint(0, 97, (2, 16), generator=g)
    out = awq.verify_rescaling(tiny, pr, res, 4, x)
    # tiny model is fp32 but verify casts scaled weights through fp16 -> small, bounded error
    assert out["runtime_division"]["max_abs_logit_diff"] < 5e-3
    assert out["merged_weights"]["max_abs_logit_diff"] < 5e-3
    pr.verify()


def test_awq_alpha0_equals_q1():
    x = torch.rand(128) + 0.1
    s = awq.awq_scales(x, 0.0)
    assert torch.equal(s, torch.ones(128, dtype=torch.float16))
    W = torch.randn(16, 128)
    weff, _ = awq.effective_weight(W, s, 4)
    assert torch.equal(weff, quant.quantize(W, 4).w_hat)


def test_apply_quantized_counts(tiny):
    pr = Pristine(tiny)
    g = torch.Generator().manual_seed(4)
    res = {"scales": {}}
    for li in range(2):
        for grp, d in (("qkv", 128), ("o", 128), ("gateup", 128), ("down", 256)):
            res["scales"][f"L{li}.{grp}.b4"] = (torch.rand(d, generator=g) + 0.5).half().numpy()
    seen = []
    info = awq.apply_quantized(tiny, res, 4, on_matrix=lambda q, qr: seen.append(q))
    assert sorted(seen) == sorted(p.qualname for p in eligible_projections(tiny))
    assert info["serialized_scale_bytes"] == 2 * 3 * 128 * 2
    pr.restore()


def test_search_and_conditions_end_to_end(tiny, tmp_path):
    import numpy as np

    from atml4.task1 import conditions
    from atml4.task1.calib import collect_input_stats, save_stats

    pr = Pristine(tiny)
    g = torch.Generator().manual_seed(5)
    batches = lambda: [torch.randint(0, 97, (2, 16), generator=torch.Generator().manual_seed(i))  # noqa: E731
                       for i in range(2)]
    res = awq.search(tiny, batches, bits_list=(4, 3), log=lambda *a: None)
    assert res["mse"].shape == (2, 4, 2, 20) and res["best_alpha_index"].shape == (2, 4, 2)
    # alpha grid contains 0 (== Q1), so the chosen MSE never exceeds the Q1 MSE
    assert (res["mse"].min(-1) <= res["mse"][..., 0] + 1e-12).all()
    awq.save_search(tmp_path / "s.npz", res)
    back = awq.load_search(tmp_path / "s.npz")
    assert set(back["scales"]) == set(res["scales"]) and back["bits"] == [4, 3]
    pr.verify()
    # condition wrappers (pruning + Q1) on CPU with tiny calibration stats
    (tmp_path / "results/task1/baseline").mkdir(parents=True)
    stats = collect_input_stats(tiny, batches())
    save_stats(tmp_path / "results/task1/baseline/calib_input_stats.npz", stats, [0, 1], 6304)
    conditions._XNORM = None
    ctx = {"device": "cpu", "pristine": pr, "root": tmp_path, "verify_pack": True}
    for name in ("p1_mag_s50", "p2_wanda_s75", "p2_wanda_2of4"):
        pr.restore()
        info = conditions.CONDITIONS[name](tiny, ctx)
        zeros = sum(int((p.module.weight == 0).sum()) for p in eligible_projections(tiny))
        assert zeros >= info["mask_pruned"] and info["mask_pruned_fraction"] == pytest.approx(
            0.5 if name != "p2_wanda_s75" else 0.75)
    pr.restore()
    info = conditions.CONDITIONS["q1_int3"](tiny, ctx)
    assert info["packing"]["roundtrip_exact"] and info["packing"]["packed_bytes"] == info["packing"]["formula_bytes"]
    pr.restore()
    conditions._XNORM = None
    np.testing.assert_equal(len(conditions.FIXED_CONDITIONS), 13)


def test_q2_outerr_unfolded_equivalent_is_exact(tiny):
    """With unquantized-equivalent scales, folded down/up weights map back to the original operators."""
    from atml4.task1 import calib

    pr = Pristine(tiny)
    g = torch.Generator().manual_seed(7)
    res = {"scales": {}}
    for li in range(2):
        for grp, d in (("qkv", 128), ("o", 128), ("gateup", 128), ("down", 256)):
            res["scales"][f"L{li}.{grp}.b4"] = (torch.rand(d, generator=g) + 0.5).half().numpy()
    W = {p.qualname: p.module.weight.detach().clone() for p in eligible_projections(tiny)}
    folded = {}
    for li in range(2):
        sd = torch.from_numpy(res["scales"][f"L{li}.down.b4"]).float()
        folded[f"model.layers.{li}.mlp.down_proj"] = W[f"model.layers.{li}.mlp.down_proj"] * sd[None, :]
        folded[f"model.layers.{li}.mlp.up_proj"] = W[f"model.layers.{li}.mlp.up_proj"] / sd[:, None]
    eq = awq.unfolded_equivalent(res, 4)
    batches = [torch.randint(0, 97, (2, 16), generator=g)]
    err = calib.relative_output_error(tiny, batches, lambda q: eq(q, folded[q]) if q in folded else None)
    assert len(err) == 4 and all(v["rel"] < 1e-10 for v in err.values())
    raw = calib.relative_output_error(tiny, batches, lambda q: folded.get(q))
    assert any(v["rel"] > 1e-3 for v in raw.values())  # the bug this guards against
    pr.verify()
