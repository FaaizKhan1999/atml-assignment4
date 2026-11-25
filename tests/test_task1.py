"""Task 1 harness tests on a tiny random Qwen2 with tied embeddings (CPU)."""
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from atml4.task1 import calib, hellaswag, ppl, storage  # noqa: E402
from atml4.task1.model import (ELIGIBLE, Pristine, eligible_projections, eligible_weight_names,  # noqa: E402
                               tie_info, unique_parameters, weights_sha256)


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    m = Qwen2ForCausalLM(cfg).eval()
    m.requires_grad_(False)
    return m


def test_enumerator(tiny):
    projs = eligible_projections(tiny)
    assert len(projs) == tiny.config.num_hidden_layers * 7
    assert [p.name for p in projs[:7]] == list(ELIGIBLE)
    names = eligible_weight_names(tiny)
    for n, _ in tiny.named_parameters():
        excluded = ("embed_tokens" in n or "norm" in n or n.endswith(".bias") or "lm_head" in n)
        assert (n in names) != excluded, n
    assert all(p.shape == tuple(p.module.weight.shape) for p in projs)


def test_tied_counted_once(tiny):
    ti = tie_info(tiny)
    assert ti["config_tie_word_embeddings"] and ti["same_data_ptr"]
    uniq = unique_parameters(tiny)
    all_numel = sum(p.numel() for _, p in tiny.named_parameters(remove_duplicate=False))
    emb = tiny.get_input_embeddings().weight.numel()
    assert sum(p.numel() for _, p in uniq) == all_numel - emb
    st, per = storage.model_storage_from_torch(tiny)
    assert st.eligible_params + st.other_params == all_numel - emb
    assert st.total_dense_fp16_bytes == 2 * st.eligible_params + 4 * st.other_params  # tiny model is fp32


def test_restoration_exact(tiny):
    pr = Pristine(tiny)
    h0 = weights_sha256(tiny)
    with torch.no_grad():
        for p in eligible_projections(tiny):
            p.module.weight.mul_(0.5)
    assert weights_sha256(tiny) != h0
    with pytest.raises(RuntimeError):
        pr.verify()
    pr.restore(verify=True)
    assert weights_sha256(tiny) == h0 == pr.sha256


def test_ppl_matches_manual(tiny):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 97, (40,), generator=g).tolist()
    lo, hi = 1, 33  # score targets 1..32 only; tokens after hi must not matter
    got = ppl.window_nll(tiny, ids, lo, hi, "cpu")
    manual = 0.0
    for t in range(lo, hi):
        logits = tiny(torch.tensor([ids[:t]])).logits[0, -1].double()
        manual -= torch.log_softmax(logits, -1)[ids[t]].item()
    assert got == pytest.approx(manual, abs=1e-4)
    rows = [{"scored_tokens": 32, "sum_nll": got}, {"scored_tokens": 8, "sum_nll": 10.0}]
    s = ppl.summarize(rows)
    assert s["N"] == 40 and s["ppl"] == pytest.approx(math.exp((got + 10.0) / 40))


def test_calib_stats_and_output_error(tiny):
    g = torch.Generator().manual_seed(2)
    recs = [{"token_ids": torch.randint(0, 97, (16,), generator=g).tolist()} for _ in range(5)]
    stats = calib.collect_input_stats(tiny, calib.calibration_batches(recs, 2, "cpu"))
    q = "model.layers.0.mlp.down_proj"
    assert stats[q]["count"] == 5 * 16 and stats[q]["sumsq"].shape == (96,)
    assert (stats[q]["sumsq"] >= 0).all()
    batches = lambda: calib.calibration_batches(recs, 2, "cpu")  # noqa: E731
    same = calib.relative_output_error(tiny, batches(), lambda name: dict(
        (p.qualname, p.module.weight.clone()) for p in eligible_projections(tiny)).get(name))
    assert all(v["num"] == 0.0 and v["rel"] == 0.0 for v in same.values()) and len(same) == 21
    w = tiny.model.layers[1].self_attn.o_proj.weight
    changed = calib.relative_output_error(tiny, batches(), lambda n: torch.zeros_like(w)
                                          if n == "model.layers.1.self_attn.o_proj" else None)
    assert list(changed) == ["model.layers.1.self_attn.o_proj"]
    assert changed["model.layers.1.self_attn.o_proj"]["rel"] == pytest.approx(1.0, rel=1e-6)


def test_hellaswag_format_matches_lm_eval():
    utils = pytest.importorskip("lm_eval.tasks.hellaswag.utils")
    datasets = pytest.importorskip("datasets")
    docs = [
        {"activity_label": "Removing ice from car", "ctx_a": "Then, the man writes over the snow.",
         "ctx_b": "then", "endings": [" , the man adds wax.", "[title] Use a scraper.  Then dry", "x [step] y",
                                      "plain ending "], "label": "3"},
        {"activity_label": "Home and Garden", "ctx_a": "[header] How to grow herbs [title] Pick a pot.",
         "ctx_b": "", "endings": ["a", "b [substeps] c", "d", "e"], "label": "0"},
    ]
    ref = list(utils.process_docs(datasets.Dataset.from_list(docs)))
    for d, r in zip(docs, ref):
        ours = hellaswag.process_doc(d)
        assert ours["query"] == r["query"] and ours["choices"] == r["choices"] and ours["gold"] == r["gold"]
        for s in d["endings"] + [d["ctx_a"]]:
            assert hellaswag.preprocess(s) == utils.preprocess(s)


def test_encode_pair_moves_trailing_space():
    class Tok:
        def encode(self, s):
            return [ord(c) for c in s]

    c, k = hellaswag.encode_pair(Tok(), "ctx  ", "x")
    assert c == [ord(x) for x in "ctx"] and k == [ord(x) for x in "  x"]


def test_acc_norm_uses_characters(tiny, monkeypatch):
    rec = {"order": 0, "source_row_index": 5, "example": {
        "activity_label": "A", "ctx_a": "b", "ctx_b": "c", "endings": ["aa", "bbbb", "c", "dddddddd"], "label": "1"}}
    lls = iter([-4.0, -6.0, -3.0, -16.0])
    monkeypatch.setattr(hellaswag, "continuation_loglik", lambda *a, **k: (next(lls), False))

    class Tok:
        def encode(self, s):
            return list(range(len(s.split())))

    r = hellaswag.score_example(tiny, Tok(), rec, "cpu")
    # raw argmax -> choice 2 (-3.0); per-char: -2.0, -1.5, -3.0, -2.0 -> choice 1
    assert r["pred_acc"] == 2 and r["pred_acc_norm"] == 1 and r["acc_norm"] == 1 and r["acc"] == 0
    assert r["char_len"] == [2, 4, 1, 8]


def test_storage_formulas():
    s = storage.sparse16_matrix_bytes(numel=10, nnz=4)
    assert s == {"mask_bytes": 2, "value_bytes": 8, "padding_bits": 6, "total_bytes": 10}
    assert storage.sparse16_matrix_bytes(16, 0)["total_bytes"] == 2
    m = storage.sparse16_model_bytes([(10, 4), (16, 16)], other_bytes=100)
    assert m["eligible_bytes"] == 10 + (2 + 32) and m["total_bytes"] == 144
    assert m["dense_fp16_eligible_bytes"] == 52 and m["nonzero_fraction"] == pytest.approx(20 / 26)
    # 4 x 300, 4-bit, g=128 -> 3 groups/row, padded to 384 cols; fp16 scale + 8-bit zero
    q = storage.quantized_matrix_bytes(4, 300, bits=4, group_size=128, scale_bits=16, zero_bits=8,
                                       pad_columns_to_group=True, joint_bitstream=False, payload_align_bits=8,
                                       metadata_align_bits=8)
    assert q["payload_bytes"] == 4 * 384 * 4 // 8 == 768
    assert q["metadata_bytes"] == 12 * 3 == 36 and q["n_groups"] == 12
    q2 = storage.quantized_matrix_bytes(3, 5, bits=3, group_size=128, scale_bits=16, zero_bits=3,
                                        pad_columns_to_group=False, joint_bitstream=False, payload_align_bits=32,
                                        metadata_align_bits=8)
    assert q2["payload_bytes"] == 8  # 45 bits -> 64
    assert q2["metadata_bytes"] == 8  # 3 groups x 19 bits = 57 -> 64
    d = storage.quantized_matrix_bytes(4, 256, bits=3)  # convention defaults: 8 groups x (128*3+3+16) bits
    assert d["total_bytes"] == -(-8 * 403 // 8) == 403 and d["n_groups"] == 8
    assert storage.two_four_matrix_bytes(16)["total_bytes"] == 18
    st = storage.model_storage([10, 16], [(5, 2), (7, 4)])
    assert st.eligible_fp16_bytes == 52 and st.other_bytes == 38 and st.total_dense_fp16_bytes == 90
