import math
import os

import pytest

torch = pytest.importorskip("torch")

from atml4.task2.kv import compress as CP  # noqa: E402
from atml4.task2.kv import kvbytes as KB  # noqa: E402
from atml4.task2.kv import kvquant as KQ  # noqa: E402
from atml4.task2.kv.kvcache import RunningQuant, forward_step, make_cache, prefill, run_suffix_and_generate  # noqa: E402
from atml4.task2.kv.prompts import extract_answer  # noqa: E402


def _tiny(attn="sdpa"):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=300, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=4096,
                      tie_word_embeddings=False, attn_implementation=attn, bos_token_id=None, eos_token_id=None)
    m = Qwen2ForCausalLM(cfg).eval().float()
    m.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def tiny():
    return _tiny()


def _ids(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 300, (n,), generator=g).tolist()


DEV = torch.device("cpu")


# ---------------------------------------------------------------- C1 mechanics
def test_split_prefill_equals_full_forward(tiny):
    ids = _ids(150)
    L = 120
    full = tiny(torch.tensor([ids])).logits[0, L:].float()
    Ks, Vs, _ = prefill(tiny, ids[:L], DEV)
    lg = forward_step(tiny, make_cache(Ks, Vs), ids[L:], L, DEV)
    assert torch.allclose(lg, full, atol=1e-4, rtol=1e-4)


def test_full_retention_reproduces_generate(tiny):
    for seed in (1, 2, 3):
        ids = _ids(100, seed)
        L = 80
        ref = tiny.generate(torch.tensor([ids]), max_new_tokens=12, do_sample=False, eos_token_id=None,
                            pad_token_id=0)[0, len(ids):].tolist()
        Ks, Vs, _ = prefill(tiny, ids[:L], DEV)
        c = CP.Compressed("fp16", Ks, Vs, torch.arange(L), torch.zeros(0, dtype=torch.uint8), 0, {})
        g = run_suffix_and_generate(tiny, make_cache(c.keys, c.values), ids[L:], L, DEV, 12, eos_ids=[])
        assert g["generated"] == ref and g["all_finite"]


def test_eviction_equals_masked_full_forward(tiny):
    """Remove context positions then append the suffix with absolute position_ids == a full forward whose 4D mask
    hides those positions from the suffix rows only (context rows plain causal, so context K/V are unchanged)."""
    ids = _ids(140, 7)
    L, S = 110, 30
    keep = torch.tensor(sorted(set(range(8)) | set(range(30, 60)) | set(range(L - 20, L))))
    n = L + S
    allowed = torch.tril(torch.ones(n, n, dtype=torch.bool))
    evicted = torch.ones(L, dtype=torch.bool)
    evicted[keep] = False
    allowed[L:, :L] &= ~evicted[None, :]
    mask4 = torch.zeros(1, 1, n, n).masked_fill(~allowed, float("-inf"))
    ref = tiny(torch.tensor([ids]), attention_mask=mask4, position_ids=torch.arange(n)[None]).logits[0, L:]
    Ks, Vs, _ = prefill(tiny, ids[:L], DEV)
    cache = make_cache([k[:, :, keep] for k in Ks], [v[:, :, keep] for v in Vs])
    assert cache.get_seq_length() == len(keep)
    got = forward_step(tiny, cache, ids[L:], L, DEV)
    assert torch.allclose(got, ref.float(), atol=1e-4, rtol=1e-4)
    # wrong positions (physical count instead of absolute) must differ -> the test is sensitive to RoPE positions
    cache2 = make_cache([k[:, :, keep] for k in Ks], [v[:, :, keep] for v in Vs])
    bad = forward_step(tiny, cache2, ids[L:], len(keep), DEV)
    assert not torch.allclose(bad, ref.float(), atol=1e-3)


def test_scores_match_eager_attention(tiny):
    from atml4.task2.kv.scoring import question_scores

    eager = _tiny("eager")
    eager.load_state_dict(tiny.state_dict())
    ids = _ids(130, 4)
    L = 100
    out = eager(torch.tensor([ids]), output_attentions=True)
    ref = torch.stack([a[0, :, L:, :L].float().mean(dim=(0, 1)) for a in out.attentions]).mean(0)
    Ks, Vs, _ = prefill(tiny, ids[:L], DEV)
    sc = question_scores(tiny, Ks, Vs, ids[L:], DEV)
    assert torch.allclose(sc["scores"], ref, atol=1e-5)


# ---------------------------------------------------------------- bytes
class _Cfg:
    num_hidden_layers, num_key_value_heads, head_dim, hidden_size, num_attention_heads = 28, 2, 128, 1536, 12


def test_m4_hand_calc():
    C = KB.channels(_Cfg)
    assert C == 28 * 2 * 128
    L = 2630  # 41 groups, r = 6
    assert KB.m_b(L, C, 4) == 2 * 7168 * (41 * 36 + 12)
    assert KB.m16(L, C) == 4 * 7168 * L
    for L in (63, 64, 65, 2608, 5246):
        assert KB.tokenwise_bytes(L, C, 128, 4) == KB.m_b(L, C, 4)


def test_n_kept_and_hybrid_never_exceed_budget():
    C = KB.channels(_Cfg)
    for L in (2608, 2631, 2652, 5196, 5246):
        B = KB.m_b(L, C, 4)
        n = KB.n_kept_max(B, C, L)
        assert KB.eviction_bytes(n, C) <= B < KB.eviction_bytes(n + 1, C)
        f = KB.hybrid_f_max(B, L, C)
        assert KB.hybrid_bytes(f, L, C)["total"] <= B
        assert all(KB.hybrid_bytes(x, L, C)["total"] > B for x in range(f + 1, L + 1))
        assert f >= 72


def test_hybrid_bytes_hand_calc():
    C = 7168
    h = KB.hybrid_bytes(300, 2630, C)
    n_low = 2330
    assert h["low_groups_per_channel"] == n_low // 64 and h["residual_fp16"] == n_low % 64
    assert h["total"] == 2 * C * (n_low // 64) * 20 + 4 * C * (300 + n_low % 64) + 2 * 300


# ---------------------------------------------------------------- quantization / packing / serialization
@pytest.mark.parametrize("bits", [2, 4])
def test_pack_roundtrip(bits):
    c = torch.randint(0, 2 ** bits, (4096,), dtype=torch.uint8)
    p = KQ.pack(c, bits)
    assert p.numel() == 4096 * bits // 8
    assert torch.equal(KQ.unpack(p, bits, 4096), c)


def test_kv_quantizer_groups_not_containing_zero():
    """Regression (deviations.md 2026-11-26): an all-positive / all-negative group must not collapse."""
    for lo, hi in ((100.0, 110.0), (-410.0, -400.0), (-3.0, 5.0), (0.0, 0.0), (7.25, 7.25)):
        x = torch.linspace(lo, hi, 64).half().float()[None]
        for bits in (2, 4):
            c, s, o, xh = KQ.quantize_groups(x, bits)
            step = float(s)
            assert float((xh - x).abs().max()) <= step / 2 + 1e-6, (lo, hi, bits)
            assert float(o) <= lo and float(o) + (2 ** bits - 1) * step >= hi  # stored grid covers the range


def test_kv_codes_use_stored_fp16_metadata():
    g = torch.Generator().manual_seed(2)
    X = (torch.randn(50, 64, generator=g) * 2.0 + 300.0).half().float()  # tight range at large magnitude (FP16 spacing 0.25)
    for bits in (2, 4):
        c, s, o, xh = KQ.quantize_groups(X, bits)
        assert s.dtype == torch.float16 and o.dtype == torch.float16
        assert torch.equal(xh, KQ.dequant_groups(c, s, o))  # dequantization uses exactly the stored values
        rel = float((xh - X).norm() / (X - X.mean(1, keepdim=True)).norm())
        assert rel < (0.6 if bits == 2 else 0.12)


def test_synthetic_channel_offset_axis():
    """Per-channel constant offsets (bias-like) + small noise: channel-wise grouping must be near-lossless and
    beat token-wise; the cache layout is [H_kv, L, d_h] and a channel group is 64 consecutive tokens of one dim."""
    g = torch.Generator().manual_seed(0)
    off = torch.empty(128).uniform_(100, 400, generator=g) * torch.where(torch.rand(128, generator=g) < .5, -1, 1)
    K = (off[None, None, :] + torch.randn(2, 256, 128, generator=g)).half()
    err = {}
    for lay in ("channel", "token"):
        q = KQ.quantize_tensor(K, lay, 4)
        err[lay] = float((q["x_hat"].float() - K.float()).norm() / K.float().norm())
    assert err["channel"] < 3e-3 < err["token"], err
    q = KQ.quantize_tensor(K, "channel", 4)
    assert q["scale"].shape == (2, 4, 128)
    _, s, o, _ = KQ.quantize_groups(K[1, 64:128, 77].float()[None], 4)
    assert torch.equal(s[0], q["scale"][1, 1, 77]) and torch.equal(o[0], q["offset"][1, 1, 77])


def test_channel_and_token_layout_groups():
    X = torch.randn(2, 128, 64)
    qc = KQ.quantize_tensor(X, "channel", 4)
    assert qc["scale"].shape == (2, 2, 64)  # (H, N/64, d)
    # each channel group = 64 consecutive tokens of one feature
    _, s, _, _ = KQ.quantize_groups(X[0, :64, 5][None], 4)
    assert torch.equal(s[0], qc["scale"][0, 0, 5])
    qt = KQ.quantize_tensor(X, "token", 4)
    assert qt["scale"].shape == (2, 128, 1)


class _TinyCfg:
    num_hidden_layers, num_key_value_heads, head_dim, hidden_size, num_attention_heads = 2, 2, 64, 256, 4


def _kv(L, seed=0):
    g = torch.Generator().manual_seed(seed)
    Ks = [(torch.randn(1, 2, L, 64, generator=g) * 2).half() for _ in range(2)]
    Vs = [torch.randn(1, 2, L, 64, generator=g).half() for _ in range(2)]
    return Ks, Vs


@pytest.mark.parametrize("L", [700, 901, 1100])
@pytest.mark.parametrize("method", CP.ALL_METHODS)
def test_serialized_length_and_roundtrip(L, method):
    Ks, Vs = _kv(L)
    scores = torch.rand(L)
    c = CP.compress(method, Ks, Vs, _TinyCfg, scores=scores)
    C = KB.channels(_TinyCfg)
    assert c.nbytes == c.formula_bytes
    if method != "fp16":
        assert c.nbytes <= KB.m_b(L, C, 4)
    if method in CP.INT4:
        assert c.nbytes == KB.m_b(L, C, 4)
    kk, vv = CP.deserialize(c, _TinyCfg, L)
    assert all(torch.equal(a, b) for a, b in zip(kk, c.keys))
    assert all(torch.equal(a, b) for a, b in zip(vv, c.values))
    if method in ("sink_recent", "qaware"):
        pos = c.positions.tolist()
        assert set(range(8)) <= set(pos) and set(range(L - 64, L)) <= set(pos)
    if method == "sink_recent":
        n = len(c.positions)
        assert c.positions.tolist() == sorted(set(range(8)) | set(range(L - (n - 8), L)))
    if method == "hybrid":
        sel = set(c.fp16_selected.tolist())
        assert set(range(8)) <= sel and set(range(L - 64, L)) <= sel
        # FP16 set + residual exactly equal to the original; quantized positions differ only within INT2 error
        fp = sel | set(c.info["residual_positions"])
        idx = torch.tensor(sorted(fp))
        assert torch.equal(c.keys[0][:, :, idx], Ks[0][:, :, idx])
        assert c.info["low_quantized_positions"] % 64 == 0
        assert c.info["low_quantized_positions"] + len(fp) == L


def test_qaware_selection_uses_scores_only():
    L = 300
    s = torch.zeros(L)
    s[[100, 150, 151]] = 1.0
    pos = CP.select_qaware(s, 75).tolist()
    assert {100, 150, 151} <= set(pos) and len(pos) == 75


def test_running_quant_groups_form_on_append():
    Ks, Vs = _kv(200)
    c = CP.compress("int4_chan", Ks, Vs, _TinyCfg)
    assert c.running["n_quantized"] == 192
    cache = make_cache(c.keys, c.values)
    rq = RunningQuant("channel", 4, 192)
    g = torch.Generator().manual_seed(5)
    new_k = [(torch.randn(1, 2, 60, 64, generator=g)).half() for _ in range(2)]
    for i, layer in enumerate(cache.layers):  # append 60 tokens -> residual 8 + 60 = 68 >= 64
        layer.keys = torch.cat([layer.keys, new_k[i]], 2)
        layer.values = torch.cat([layer.values, new_k[i]], 2)
    raw = [layer.keys[0, :, 192:256].clone() for layer in cache.layers]
    rq(cache)
    assert rq.n_q == 256 and rq.groups_quantized_during_decode == 1
    for layer, r in zip(cache.layers, raw):
        assert torch.equal(layer.keys[0, :, 192:256], KQ.quantize_tensor(r, "channel", 4)["x_hat"])
        assert torch.equal(layer.keys[0, :, 256:], new_k[cache.layers.index(layer)][0, :, 64 - 8:])
    rq(cache)
    assert rq.n_q == 256  # 4 FP16 tokens remain, no new group
    assert KB.live_quantized_bytes(256, 4, KB.channels(_TinyCfg)) == \
        2 * 256 * (256 // 64) * 36 + 4 * 256 * 4


# ---------------------------------------------------------------- extractor / prompts
def test_extractor():
    assert extract_answer("26836") == "26836"
    assert extract_answer("The code is 26836.") == "26836"
    assert extract_answer("123456 then 54321") == "54321"  # 6-digit run is not exactly five
    assert extract_answer("1234 12 3") is None
    assert extract_answer("a12345b67890") == "12345"
    assert extract_answer("") is None


@pytest.mark.skipif(not os.path.isdir(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct")),
                    reason="pinned tokenizer not cached")
def test_all_prompts_split_and_evidence():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    from atml4.common.meta import model_revision
    from atml4.task2.kv import prompts as PR

    mid, rev = model_revision("target")
    tok = AutoTokenizer.from_pretrained(mid, revision=rev)
    cases = PR.load_cases()
    assert len(cases) == 180
    for case in cases:
        p = PR.build(case, tok)
        assert tok.decode(p.suffix_ids).startswith("What is the access code for CASE-")
        assert tok.decode(p.suffix_ids).endswith("<|im_end|>\n<|im_start|>assistant\n")
        ev = tok.decode([p.ids[i] for i in p.evidence_tokens])
        assert case["evidence_text"] in ev and len(ev) < len(case["evidence_text"]) + 12
        assert tok.decode(p.answer_ids[:-1]) == case["code"] and p.answer_ids[-1] == 151645
        assert p.L < 65536 and math.isfinite(p.L)


def test_hybrid_infeasible_short_context_raises():
    C = KB.channels(_TinyCfg)
    with pytest.raises(ValueError):
        KB.hybrid_f_max(KB.m_b(768, C, 4), 768, C)


def test_all_real_lengths_feasible():
    C = KB.channels(_Cfg)
    for L in list(range(2600, 2660)) + list(range(5190, 5250)):
        B = KB.m_b(L, C, 4)
        assert KB.n_kept_max(B, C, L) >= 72 and KB.hybrid_f_max(B, L, C) >= 72


def test_evaluate_prompt_end_to_end_tiny():
    from atml4.task2.kv.evaluate import evaluate_prompt

    tiny = _tiny().half()
    from atml4.task2.kv.prompts import Prompt

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(str(i % 10) for i in ids)

        def convert_tokens_to_ids(self, t):
            return 299

    ids = _ids(1130, 11)
    L = 1100
    p = Prompt("X", {"target_tokens": 2048, "position": "start", "distractors": 0, "code": "12345"}, ids, L,
               ids[L:], list(range(40, 52)), [1, 2, 3, 4, 5, 299])
    rec = evaluate_prompt(tiny, Tok(), p, DEV)
    C = KB.channels(tiny.config)
    for m, r in rec["methods"].items():
        assert r["all_finite"] and math.isfinite(r["ref_nll_sum"])
        assert r["packed_bytes"] <= (KB.m16(L, C) if m == "fp16" else KB.m_b(L, C, 4))
    assert rec["methods"]["fp16"]["kl_first_token_vs_fp16"] == pytest.approx(0, abs=1e-9)
    # FP16 random tiny model: split vs one-shot forward differ by rounding and near-ties can flip late tokens;
    # exact token equality is tested in FP32 above and on the real model in the GPU validation kernel.
    assert rec["methods"]["fp16"]["generated_ids"][:8] == tiny.generate(
        torch.tensor([ids]), max_new_tokens=8, do_sample=False, eos_token_id=299, pad_token_id=0)[0, len(ids):].tolist()
    assert rec["methods"]["int4_chan"]["packed_bytes"] == KB.m_b(L, C, 4) == rec["methods"]["int4_tok"]["packed_bytes"]
    r = rec["methods"]["int4_chan"]
    assert r["recon_error"]["K"]["mean"] > 0 and r["evidence_retained_fp16"] is None


def test_bench_encode_decode_roundtrip():
    import importlib.util

    from atml4.common.meta import repo_root

    spec = importlib.util.spec_from_file_location("t2crun", repo_root() / "scripts/task2c_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = torch.Generator().manual_seed(3)
    Ks = [torch.randn(4, 200, 64, generator=g).half() for _ in range(2)]
    Vs = [torch.randn(4, 200, 64, generator=g).half() for _ in range(2)]
    qpos = torch.tensor(sorted(torch.randperm(200, generator=g)[:128].tolist()))
    for kind, N in (("int4_chan", 192), ("int4_tok", 192), ("int2_hybrid_low", 128)):
        blobs = mod._encode(Ks, Vs, kind, qpos, 192)
        dec = mod._decode(blobs, (4, N, 64), kind)
        for (k, v), K, V in zip(dec, Ks, Vs):
            src_k = K[:, qpos] if kind == "int2_hybrid_low" else K[:, :192]
            src_v = V[:, qpos] if kind == "int2_hybrid_low" else V[:, :192]
            bits = 2 if kind == "int2_hybrid_low" else 4
            kl = "token" if kind == "int4_tok" else "channel"
            assert torch.equal(k, KQ.quantize_tensor(src_k, kl, bits)["x_hat"])
            assert torch.equal(v, KQ.quantize_tensor(src_v, "token", bits)["x_hat"])


def test_zp_form_supplementary():
    # group containing 0: identical to the original (clamped) Task 1 form; groups not containing 0: no collapse
    x = torch.tensor([[-3.0, -1.0, 0.0, 2.0, 5.0] + [1.0] * 59, list(torch.linspace(100, 110, 64))])
    c, s, f, xh, nfb = KQ.quantize_groups_zp(x, 4)
    assert nfb == 0 and float(s.min()) > 0
    assert float((xh[1] - x[1]).abs().max()) <= float(s[1]) / 2 + 1e-4  # no collapse
    assert torch.equal(KQ.dequant_groups_zp(c, s, f), xh)
    # |z| > 2048 -> min/offset fallback flagged by a negative stored scale
    y = (torch.linspace(0, 0.01, 64) + 1000.0)[None]
    c, s, f, xh, nfb = KQ.quantize_groups_zp(y, 4)
    assert nfb == 1 and float(s[0]) < 0
    assert torch.equal(KQ.dequant_groups_zp(c, s, f), xh) and float((xh - y).abs().max()) < 0.01


def test_zp_serialization_roundtrip_with_fallback():
    g = torch.Generator().manual_seed(1)
    X = torch.randn(2, 128, 64, generator=g).half()
    X[0, :64, 3] = 1000.0 + torch.rand(64, generator=g).half() * 0.01  # forces a fallback channel group
    q = KQ.quantize_tensor(X, "channel", 4, "zp")
    assert q["zp_fallback_groups"] >= 1
    buf = torch.cat(KQ.serialize_quant(q))
    xh, off = KQ.deserialize_quant(buf, 0, (2, 128, 64), "channel", 4, "zp")
    assert off == buf.numel() and torch.equal(xh, q["x_hat"])
