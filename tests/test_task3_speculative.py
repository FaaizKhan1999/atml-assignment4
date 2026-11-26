import hashlib

import pytest

torch = pytest.importorskip("torch")

from atml4.task3 import toy  # noqa: E402
from atml4.task3.speculative import HFLM, ToyMarkov, autoregressive_generate, speculative_generate  # noqa: E402


class HistoryLM:
    """Next-token law depends on the WHOLE cached history (hash) -> any cache rollback error changes the law."""

    def __init__(self, V=6, salt="t", sharp=3.0):
        self.V, self.salt, self.sharp = V, salt, sharp
        self.reset()

    def reset(self):
        self.cached, self.passes = [], 0

    def _logits(self, hist):
        h = hashlib.sha256((self.salt + ",".join(map(str, hist))).encode()).digest()
        g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little"))
        return torch.randn(self.V, generator=g) * self.sharp

    def feed(self, tokens):
        rows = []
        for t in tokens:
            self.cached.append(t)
            rows.append(self._logits(self.cached))
        self.passes += 1
        return torch.stack(rows)

    def crop(self, n):
        self.cached = self.cached[:n]


# ---------------------------------------------------------------- chi-square (PDF Step 1)
@pytest.mark.parametrize("gamma", [1, 2, 3])
def test_chi_square_correct_sampler_not_rejected(gamma):
    r = toy.run(100_000, gamma, 6304 + gamma)
    cs = toy.chi_square(r["counts"], 100_000)
    assert cs["min_expected"] >= 10
    assert cs["p_value"] >= 0.01, cs


@pytest.mark.parametrize("gamma", [1, 2, 3])
def test_chi_square_negative_control_rejected(gamma):
    r = toy.run(100_000, gamma, 6304 + gamma, residual="p")
    assert toy.chi_square(r["counts"], 100_000)["p_value"] < 0.01


def test_exact_joint_sums_to_one():
    assert abs(sum(toy.exact_joint().values()) - 1) < 1e-12
    assert min(toy.exact_joint().values()) * 100_000 >= 10


# ---------------------------------------------------------------- mechanics
def test_identical_history_models_accept_everything():
    """draft == target with a history-dependent law: acceptance must be exactly 1 over many rounds, which only
    holds if both caches are rolled back / advanced to exactly the returned continuation."""
    for gamma in (1, 2, 4):
        g = torch.Generator().manual_seed(gamma)
        r = speculative_generate(HistoryLM(salt="a"), HistoryLM(salt="a"), [1, 2, 3], 40, gamma, g)
        s = r["stats"]
        assert s["accepted"] == s["proposed"] == s["examined"] and s["residual"] == 0
        assert len(r["tokens"]) == 40


def test_rollback_with_rejections_matches_target_law_per_step():
    """Different history-dependent target/draft: the internal assertions check cache == returned prefix after
    every round; acceptances < proposals so rollbacks happen."""
    g = torch.Generator().manual_seed(0)
    r = speculative_generate(HistoryLM(salt="t"), HistoryLM(salt="d"), [0], 60, 4, g)
    assert r["stats"]["residual"] > 0 and len(r["tokens"]) == 60


def test_bonus_path_and_one_round():
    P = torch.tensor([[0.5, 0.5, 0.0], [0.3, 0.3, 0.4], [0.2, 0.2, 0.6]], dtype=torch.float64) + 0.0
    P = (P + 0.05) / (P + 0.05).sum(1, keepdim=True)
    pi = torch.tensor([0.3, 0.3, 0.4], dtype=torch.float64)
    t, d = ToyMarkov(P, pi), ToyMarkov(P, pi)
    r = speculative_generate(t, d, [3], 3, 2, torch.Generator().manual_seed(1))
    s = r["stats"]
    assert s["rounds"] == 1 and s["bonus"] == 1 and s["accepted"] == 2 and s["target_passes"] == 1
    assert len(r["tokens"]) == 3


def test_budget_truncation_and_one_token_round():
    P = torch.full((3, 3), 1 / 3, dtype=torch.float64)
    t, d = ToyMarkov(P, P[0]), ToyMarkov(P, P[0])
    r = speculative_generate(t, d, [3], 6, 4, torch.Generator().manual_seed(2))
    s = r["stats"]
    assert len(r["tokens"]) == 6  # round 1: gamma 4 -> 5 tokens; round 2: remaining 1 -> one-token round
    assert s["one_token_rounds"] == 1 and s["proposed"] == 4 and s["rounds"] == 2
    r = speculative_generate(t, d, [3], 1, 4, torch.Generator().manual_seed(2))
    assert len(r["tokens"]) == 1 and r["stats"]["proposed"] == 0


def test_eos_mid_proposal_stops_immediately():
    # deterministic chain BOS -> 1 -> 0 (EOS) -> 2 ...; draft == target so EOS is an ACCEPTED proposal
    P = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    pi = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    t, d = ToyMarkov(P, pi), ToyMarkov(P, pi)
    r = speculative_generate(t, d, [3], 10, 4, torch.Generator().manual_seed(3), eos_ids=[0])
    assert r["tokens"] == [1, 0] and r["stop"] == "eos"
    assert r["stats"]["examined"] == 2 and r["stats"]["bonus"] == 0
    # EOS via the bonus token
    r = speculative_generate(t, d, [3], 10, 1, torch.Generator().manual_seed(3), eos_ids=[0])
    assert r["tokens"] == [1, 0] and r["stats"]["bonus"] == 1


def test_eos_via_residual():
    # target always emits EOS(0); draft never proposes 0 -> first proposal rejected, residual = EOS
    Pt = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    Pd = torch.tensor([[0, 0.5, 0.5], [0, 0.5, 0.5], [0, 0.5, 0.5]], dtype=torch.float64)
    r = speculative_generate(ToyMarkov(Pt, Pt[0]), ToyMarkov(Pd, Pd[0]), [3], 10, 3,
                             torch.Generator().manual_seed(4), eos_ids=[0])
    assert r["tokens"] == [0] and r["stop"] == "eos" and r["stats"]["residual"] == 1 and r["stats"]["examined"] == 1


# ---------------------------------------------------------------- tiny HF models
def _tiny(seed):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      attn_implementation="sdpa")
    return Qwen2ForCausalLM(cfg).eval().float().requires_grad_(False)


def test_hf_identical_models_full_acceptance_and_cache_positions():
    m = _tiny(0)
    dev = torch.device("cpu")
    r = speculative_generate(HFLM(m, dev), HFLM(m, dev), [1, 5, 9, 2], 30, 3, torch.Generator().manual_seed(0))
    s = r["stats"]
    assert s["accepted"] == s["proposed"] and len(r["tokens"]) == 30


def test_hf_split_prefill_equals_full_forward_and_rollback():
    tm, dm = _tiny(0), _tiny(1)
    dev = torch.device("cpu")
    T, D = HFLM(tm, dev), HFLM(dm, dev)
    prompt = [1, 5, 9, 2, 7]
    r = speculative_generate(T, D, prompt, 25, 4, torch.Generator().manual_seed(5))
    seq = prompt + r["tokens"]
    assert r["stats"]["residual"] > 0
    # the target cache after generation must equal a fresh full forward of seq[:-1]
    full = tm(torch.tensor([seq])).logits[0].float()
    last = T.feed([seq[-1]])[-1]
    assert torch.allclose(last, full[-1], atol=1e-4)
    k_cache = T.cache.layers[0].keys
    T2 = HFLM(tm, dev)
    T2.feed(seq)
    assert torch.allclose(k_cache, T2.cache.layers[0].keys, atol=1e-5)


def test_autoregressive_reference_runs():
    m = _tiny(0)
    r = autoregressive_generate(HFLM(m, torch.device("cpu")), [1, 2, 3], 10, torch.Generator().manual_seed(0))
    assert len(r["tokens"]) == 10 and r["target_passes"] == 10


def test_batched_sample_rows_match_single_row_runs():
    from atml4.task3.generate import batched_sample

    m = _tiny(0)
    dev = torch.device("cpu")
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8, 9], [10]]
    caps, seeds = [12, 5, 9], [11, 22, 33]
    batch = batched_sample(m, prompts, caps, seeds, eos_ids=[63], device=dev, pad_id=0)
    for p, c, s, b in zip(prompts, caps, seeds, batch):
        single = batched_sample(m, [p], [c], [s], eos_ids=[63], device=dev, pad_id=0)[0]
        assert single["tokens"] == b["tokens"] and len(b["tokens"]) <= c
        ar = autoregressive_generate(HFLM(m, dev), p, c, torch.Generator().manual_seed(s), eos_ids=[63])
        assert ar["tokens"] == b["tokens"]  # same per-row seed -> same draws as the batch-1 reference sampler
