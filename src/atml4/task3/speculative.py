"""Exact speculative sampling (Leviathan et al., 2023) over a small model interface shared by toy and HF models.

Model interface (batch 1; padding never occurs and is asserted):
  feed(tokens) -> logits [n, V] FP32 for the next-token distribution after each fed token; appends to the cache
  crop(n)      -> keep the first n cached tokens
  cached       -> list of cached token ids (the exact cache content, used to check consistency)

Cache invariant at the start of every round: each model's cache holds a prefix of the current sequence `seq` of
length <= len(seq) - 1; the tokens seq[len(cache):] are fed first (the last emitted token, plus any token the draft
never processed, e.g. x_gamma after a fully accepted round), so cached token lists always equal a prefix of the
returned continuation and absolute positions follow from the cache length.

Round (gamma_eff = min(gamma, remaining - 1); remaining = new-token budget left):
  1. draft: q_1 = draft.feed(seq[len(draft):])[-1]; x_1 ~ q_1; q_i = draft.feed([x_{i-1}])[-1]; x_i ~ q_i
     (gamma_eff draft forward passes; full q vectors stored)
  2. target: ONE pass over seq[len(target):] + [x_1..x_gamma_eff] -> p_1..p_{gamma_eff+1}
  3. for i = 1..gamma_eff: u ~ U(0,1); accept x_i iff u < min(1, p_i(x_i)/q_i(x_i)); stop after an accepted EOS
  4. first rejection at i: y ~ normalize([p_i - q_i]+) (negative control: y ~ p_i); emitted = x_1..x_{i-1}, y
  5. all accepted: bonus y ~ p_{gamma_eff+1}; emitted = x_1..x_gamma_eff, y
  6. emitted is cut after the first EOS; then target.crop(len(seq_new) - 1) and draft.crop(min(len(draft),
     len(seq_new) - 1)) (draft keeps only tokens consistent with seq_new).
With remaining == 1, gamma_eff = 0: a one-token round samples directly from p_1 (no draft work). All probabilities are
FP32 softmax(logits / T) with T = 1, full vocabulary, no top-k / top-p.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch


def probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample(p: torch.Tensor, gen: torch.Generator) -> int:
    return int(torch.multinomial(p, 1, generator=gen))


def uniform(gen: torch.Generator, device) -> float:
    return float(torch.rand(1, generator=gen, device=device))


class ToyMarkov:
    """Markov LM over V symbols + a BOS id (= V): next-token law depends on the last token only."""

    def __init__(self, P: torch.Tensor, pi0: torch.Tensor):
        V = P.shape[0]
        self.V = V
        T = torch.zeros(V + 1, V + 1, dtype=torch.float64)
        T[:V, :V] = P
        T[V, :V] = pi0
        self.logT = torch.log(T).float()  # BOS column = -inf (never generated)
        self.cached: list[int] = []
        self.passes = 0

    def feed(self, tokens):
        assert len(tokens) >= 1
        self.cached += list(tokens)
        self.passes += 1
        return self.logT[torch.tensor(tokens)]

    def crop(self, n):
        assert 0 <= n <= len(self.cached)
        self.cached = self.cached[:n]

    def reset(self):
        self.cached, self.passes = [], 0


class HFLM:
    """HF causal LM with a DynamicCache; batch 1."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.reset()

    def reset(self):
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.cached: list[int] = []
        self.passes = 0

    @torch.no_grad()
    def feed(self, tokens):
        assert len(tokens) >= 1
        n0 = len(self.cached)
        assert self.cache.get_seq_length() == n0, "cache length out of sync"
        x = torch.tensor([list(tokens)], device=self.device)
        assert x.shape[0] == 1  # batch 1: no padding ever
        pos = torch.arange(n0, n0 + len(tokens), device=self.device)
        out = self.model(input_ids=x, past_key_values=self.cache, use_cache=True, cache_position=pos,
                         position_ids=pos[None])
        self.cached += list(tokens)
        self.passes += 1
        return out.logits[0].float()

    def crop(self, n):
        assert 0 <= n <= len(self.cached)
        self.cache.crop(n)
        self.cached = self.cached[:n]
        assert self.cache.get_seq_length() == n


@dataclass
class Stats:
    rounds: int = 0
    proposed: int = 0
    examined: int = 0
    accepted: int = 0
    target_passes: int = 0
    draft_passes: int = 0
    bonus: int = 0
    residual: int = 0
    one_token_rounds: int = 0
    accepted_prefix: list = field(default_factory=list)
    timing: dict = field(default_factory=lambda: {"draft_s": 0.0, "verify_s": 0.0, "accept_s": 0.0, "cache_s": 0.0})

    def as_dict(self):
        d = dict(self.__dict__)
        d["accepted_over_examined"] = self.accepted / self.examined if self.examined else None
        d["accepted_over_proposed"] = self.accepted / self.proposed if self.proposed else None
        d["mean_accepted_prefix"] = sum(self.accepted_prefix) / len(self.accepted_prefix) if self.accepted_prefix else None
        return d


def _sync(device):
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def speculative_generate(target, draft, prompt: list[int], max_new: int, gamma: int, gen: torch.Generator,
                         eos_ids=(), residual: str = "positive", temperature: float = 1.0, timed: bool = False,
                         device=None) -> dict:
    """Returns {'tokens', 'stop', 'stats', 'prefill_s'}. stop in {'eos', 'cap'}."""
    assert residual in ("positive", "p")  # "p" = deliberately INCORRECT negative control
    eos = set(eos_ids)
    dev = device if device is not None else torch.device("cpu")
    st = Stats()
    t0 = time.perf_counter()
    for m in (target, draft):
        m.reset()
        if len(prompt) > 1:
            m.feed(prompt[:-1])  # prefill; the last prompt token is fed with the first round
    if timed:
        _sync(device)
    prefill_s = time.perf_counter() - t0
    tpass0, dpass0 = target.passes, draft.passes
    seq = list(prompt)
    out: list[int] = []
    stop = "cap"
    while len(out) < max_new:
        remaining = max_new - len(out)
        g = min(gamma, remaining - 1)
        st.rounds += 1
        t = time.perf_counter()
        xs, qs = [], []
        if g > 0:
            ql = draft.feed(seq[len(draft.cached):])[-1]
            for i in range(g):
                if i > 0:
                    ql = draft.feed([xs[-1]])[-1]
                q = probs(ql, temperature)
                qs.append(q)
                xs.append(sample(q, gen))
        else:
            st.one_token_rounds += 1
        if timed:
            _sync(device)
        t1 = time.perf_counter()
        pl = target.feed(seq[len(target.cached):] + xs)[-(g + 1):]
        if timed:
            _sync(device)
        t2 = time.perf_counter()
        emitted, k, rejected = [], 0, False
        for i in range(g):
            p = probs(pl[i], temperature)
            x = xs[i]
            a = min(1.0, float(p[x]) / float(qs[i][x]))
            if uniform(gen, dev) < a:
                emitted.append(x)
                k += 1
                if x in eos:
                    break
            else:
                rejected = True
                if residual == "positive":
                    r = torch.clamp(p - qs[i], min=0)
                    z = r.sum()
                    assert float(z) > 0, "rejection with p <= q everywhere is impossible"
                    r = r / z
                else:
                    r = p
                emitted.append(sample(r, gen))
                st.residual += 1
                break
        if not rejected and not (emitted and emitted[-1] in eos):
            emitted.append(sample(probs(pl[g], temperature), gen))
            if g > 0:
                st.bonus += 1
        st.proposed += g
        st.examined += (k + 1) if rejected else k
        st.accepted += k
        if g > 0:
            st.accepted_prefix.append(k)
        if timed:
            _sync(device)
        t3 = time.perf_counter()
        cut = next((j + 1 for j, y in enumerate(emitted) if y in eos), None)
        if cut is not None:
            emitted = emitted[:cut]
        emitted = emitted[:remaining]
        out += emitted
        seq += emitted
        target.crop(len(seq) - 1)
        keep = 0
        dc = draft.cached
        lim = min(len(dc), len(seq) - 1)
        while keep < lim and dc[keep] == seq[keep]:
            keep += 1
        draft.crop(keep)
        assert target.cached == seq[:-1] and draft.cached == seq[:keep]
        if timed:
            _sync(device)
        t4 = time.perf_counter()
        st.timing["draft_s"] += t1 - t
        st.timing["verify_s"] += t2 - t1
        st.timing["accept_s"] += t3 - t2
        st.timing["cache_s"] += t4 - t3
        if cut is not None:
            stop = "eos"
            break
    st.target_passes = target.passes - tpass0
    st.draft_passes = draft.passes - dpass0
    return {"tokens": out, "stop": stop, "stats": st.as_dict(), "prefill_s": prefill_s}


def autoregressive_generate(target, prompt: list[int], max_new: int, gen: torch.Generator, eos_ids=(),
                            temperature: float = 1.0, timed: bool = False, device=None) -> dict:
    """Target-only reference sampler with the same interface and cache discipline."""
    eos = set(eos_ids)
    t0 = time.perf_counter()
    target.reset()
    logits = target.feed(prompt)[-1]
    if timed:
        _sync(device)
    prefill_s = time.perf_counter() - t0
    out, stop = [], "cap"
    t = time.perf_counter()
    while len(out) < max_new:
        y = sample(probs(logits, temperature), gen)
        out.append(y)
        if y in eos:
            stop = "eos"
            break
        if len(out) < max_new:
            logits = target.feed([y])[-1]
    if timed:
        _sync(device)
    return {"tokens": out, "stop": stop, "prefill_s": prefill_s, "decode_s": time.perf_counter() - t,
            "target_passes": target.passes}
