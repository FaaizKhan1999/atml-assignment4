import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from atml4.task3 import distill as DS  # noqa: E402


def _tiny(seed, vocab=200):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      attn_implementation="sdpa", tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg).eval().float().requires_grad_(False)


class Tok:
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True):
        t = msgs[0]["content"]
        return [1] + [3 + (ord(c) % 150) for c in t[:12]] + [2]


def _prompts():
    return {"math": [{"id": f"m{i}", "text": f"math question {i}", "category": "math"} for i in range(5)],
            "short": [{"id": f"s{i}", "text": f"short {i}", "category": "short"} for i in range(3)]}


def test_lora_init_seeded_and_trainable_only_adapters():
    a = DS.add_lora(_tiny(1), 6304)
    b = DS.add_lora(_tiny(1), 6304)
    c = DS.add_lora(_tiny(1), 6305)
    na = {n: p for n, p in a.named_parameters() if "lora_A" in n}
    nb = {n: p for n, p in b.named_parameters() if "lora_A" in n}
    nc = {n: p for n, p in c.named_parameters() if "lora_A" in n}
    assert all(torch.equal(na[k], nb[k]) for k in na) and not all(torch.equal(na[k], nc[k]) for k in na)
    assert all(float(p.abs().sum()) == 0 for n, p in a.named_parameters() if "lora_B" in n)
    pc = DS.param_counts(a)
    assert pc["trainable_adapter"] == 2 * (8 * (64 + 64) * 2 + 8 * (64 + 32) * 2)  # 2 layers: q,o (64x64); k,v (64->32)


@pytest.mark.parametrize("method", ["off", "on"])
def test_update_accounting_straddle_and_quota(method):
    dev = torch.device("cpu")
    teacher, student = _tiny(0), DS.add_lora(_tiny(1), 6304)
    prompts = _prompts()
    coll = None
    if method == "off":
        g = torch.Generator().manual_seed(0)
        coll = {p["id"]: {"tokens": torch.randint(3, 199, (int(torch.randint(5, 30, (1,), generator=g)),),
                                                   generator=g).tolist(), "stop": "cap", "seed": 0}
                for c in prompts.values() for p in c}
    tr = DS.Trainer(method, 6304, teacher, student, Tok(), prompts, dev, collection=coll,
                    quota={"math": 70, "short": 25}, tokens_per_update=16, log=lambda *a: None)
    res = tr.train()
    assert res["scored"] == {"math": 70, "short": 25}
    toks = [u["tokens"] for u in tr.updates]
    assert sum(toks) == 95 and all(t == 16 for t in toks[:-1]) and toks[-1] == 95 - 16 * 5
    assert [u["normaliser"] for u in tr.updates] == toks
    assert sum(p["scored_tokens"] for p in tr.presentations) == 95
    # straddling: some presentation is split across two updates
    ks = [c["k"] for u in tr.updates for c in u["chunks"]]
    assert len(ks) > len(set(ks))
    # quota truncation recorded
    assert any(p["termination"] == "quota" for p in tr.presentations) or \
        all(p["scored_tokens"] == p["response_len"] for p in tr.presentations)
    # interleaving keeps categories proportional
    assert len(tr.updates) == 6 and res["skipped_updates"] == 0
    assert any(p["category"] == "short" for p in tr.presentations[:4])


def test_kl_objective_matches_manual():
    dev = torch.device("cpu")
    teacher, student = _tiny(0), _tiny(1)
    prompt, resp = [1, 5, 6, 2], [7, 8, 9, 10, 11]
    tlp = DS.teacher_logprobs(teacher, prompt, resp, dev)
    kl = DS.student_kl_sum(student, prompt, resp, 1, 4, tlp, dev)
    full_t = torch.log_softmax(teacher(torch.tensor([prompt + resp])).logits[0].float(), -1)
    full_s = torch.log_softmax(student(torch.tensor([prompt + resp])).logits[0].float(), -1)
    man = sum(float((full_t[3 + j].exp() * (full_t[3 + j] - full_s[3 + j])).sum()) for j in range(1, 4))
    assert abs(float(kl) - man) < 1e-4


def test_training_reduces_kl_and_merge_matches_unmerged(tmp_path):
    dev = torch.device("cpu")
    teacher = _tiny(0)
    student = DS.add_lora(_tiny(1), 6304)
    prompt, resp = [1, 5, 6, 2], list(range(10, 40))
    tlp = DS.teacher_logprobs(teacher, prompt, resp, dev)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-2)
    k0 = float(DS.student_kl_sum(student, prompt, resp, 0, 30, tlp, dev))
    for _ in range(30):
        opt.zero_grad()
        DS.student_kl_sum(student, prompt, resp, 0, 30, tlp, dev).backward()
        opt.step()
    k1 = float(DS.student_kl_sum(student, prompt, resp, 0, 30, tlp, dev))
    assert k1 < k0
    student.save_pretrained(tmp_path / "adapter")
    merged = DS.merge_adapter(_tiny(1), tmp_path / "adapter")
    x = torch.tensor([prompt + resp])
    with torch.no_grad():
        a = student(input_ids=x).logits
        b = merged(input_ids=x).logits
    assert torch.allclose(a, b, atol=1e-4)
