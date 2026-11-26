"""Off-/on-policy forward-KL distillation of the 0.5B student into the 1.5B teacher (PDF Task 3 Steps 3-4).

Fixed settings (PDF table): LoRA r=8, alpha=16, dropout 0 on q/k/v/o_proj (peft; adapters FP32 via
autocast_adapter_dtype, backbone FP16 frozen); FP16 autocast + dynamic GradScaler, unscale before global-norm
clipping at 1.0; AdamW LR 3e-4 (constant: no schedule is specified), weight decay 0.01; microbatch 1; one optimizer
update per EXACTLY 256 teacher-scored response tokens (short final update permitted); quota 40,000 scored tokens =
30,000 math + 10,000 short; response caps 192 / 32; temperature 1, top-p 1.

Objective: per update, (1 / N_update) * sum over its supervised response prefixes h of KL(p_T(.|h) || q_phi(.|h)) over
the full 151,936-logit vocabulary, FP32 log-softmax of both models' logits. The sampled token is never a hard target.
A response straddling an update boundary is split: its cached teacher log-probs are reused and the student forward
is recomputed (with the updated weights) for the remainder. When a category quota is reached mid-response only the
permitted prefix is scored (termination 'quota').

Schedule (identical rule for both methods and both seeds): within each category a fixed seeded permutation
(random.Random('PA4-T3-schedule:<category>'), independent of the training seed), cycled round-robin with reuse;
categories interleaved by always drawing from the category furthest behind its quota proportionally (scored/quota;
ties -> math). Off-policy: the response is the frozen teacher-collection response of that prompt. On-policy: a fresh
batch-1 sample from the CURRENT student (rollout seed = training_seed * 1_000_003 + presentation index), not
differentiated through.
"""
from __future__ import annotations

import random
import time

import torch

from .data import CAPS_TRAIN, EOS_IDS, chat_ids
from .generate import batched_sample

QUOTA = {"math": 30_000, "short": 10_000}
TOKENS_PER_UPDATE = 256
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])


def schedule_order(prompts: dict[str, list[dict]]) -> dict[str, list[dict]]:
    out = {}
    for cat, ps in prompts.items():
        idx = list(range(len(ps)))
        random.Random(f"PA4-T3-schedule:{cat}").shuffle(idx)
        out[cat] = [ps[i] for i in idx]
    return out


def add_lora(student, seed: int):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)  # same LoRA init rule per seed (B = 0, A kaiming-uniform from this seed)
    cfg = LoraConfig(bias="none", task_type="CAUSAL_LM", **LORA)
    m = get_peft_model(student, cfg, autocast_adapter_dtype=True)
    for n, p in m.named_parameters():
        if "lora_" in n:
            assert p.requires_grad and p.dtype == torch.float32, n
        else:
            assert not p.requires_grad, n
    return m


def param_counts(m) -> dict:
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {"total": tot, "trainable_adapter": tr, "frozen_backbone": tot - tr}


@torch.no_grad()
def teacher_logprobs(teacher, prompt: list[int], resp: list[int], device) -> torch.Tensor:
    """FP32 log-probs [n, V] of the teacher at the n response prefixes (positions predicting resp[0..n-1])."""
    x = torch.tensor([prompt + resp], device=device)
    lg = teacher(input_ids=x, use_cache=False).logits[0, len(prompt) - 1:len(prompt) + len(resp) - 1]
    return torch.log_softmax(lg.float(), -1)


def student_kl_sum(student, prompt, resp, lo: int, hi: int, t_logp: torch.Tensor, device) -> torch.Tensor:
    """sum_{j in [lo, hi)} KL(p_T || q) at response prefixes j (student forward over prompt + resp[:hi])."""
    x = torch.tensor([prompt + resp[:hi]], device=device)
    with torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.float16,
                        enabled=device.type == "cuda"):
        lg = student(input_ids=x, use_cache=False).logits[0, len(prompt) - 1 + lo:len(prompt) - 1 + hi]
    lq = torch.log_softmax(lg.float(), -1)
    lp = t_logp[lo:hi]
    return (lp.exp() * (lp - lq)).sum()


class Trainer:
    def __init__(self, method: str, seed: int, teacher, student, tok, prompts, device, collection=None,
                 log=print, quota=None, tokens_per_update=TOKENS_PER_UPDATE):
        assert method in ("off", "on")
        self.method, self.seed, self.dev, self.tok, self.log = method, seed, device, tok, log
        self.teacher, self.student = teacher, student
        self.quota = dict(quota or QUOTA)
        self.tpu = tokens_per_update
        self.order = schedule_order(prompts)
        self.next_idx = {c: 0 for c in self.order}
        self.collection = collection  # {prompt_id: {"tokens", "stop", "seed"}}
        self.scored = {c: 0 for c in self.order}
        params = [p for p in student.parameters() if p.requires_grad]
        self.params = params
        self.opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        self.presentations, self.updates = [], []
        self.time = {"student_rollout_s": 0.0, "teacher_scoring_s": 0.0, "fwd_bwd_opt_s": 0.0}

    def _sync(self):
        if self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    def _next_category(self):
        open_c = [c for c in ("math", "short") if self.scored[c] < self.quota[c]]
        if not open_c:
            return None
        return min(open_c, key=lambda c: (self.scored[c] / self.quota[c], 0 if c == "math" else 1))

    def _next_presentation(self, k: int):
        cat = self._next_category()
        if cat is None:
            return None
        ps = self.order[cat]
        pr = ps[self.next_idx[cat] % len(ps)]
        self.next_idx[cat] += 1
        prompt = chat_ids(self.tok, pr["text"])
        if self.method == "off":
            c = self.collection[pr["id"]]
            resp, stop, rseed = list(c["tokens"]), c["stop"], c["seed"]
        else:
            rseed = self.seed * 1_000_003 + k
            t = time.perf_counter()
            self.student.eval()
            r = batched_sample(self.student, [prompt], [CAPS_TRAIN[cat]], [rseed], EOS_IDS, self.dev,
                               pad_id=self.tok.pad_token_id)[0]
            self.student.train()
            self._sync()
            self.time["student_rollout_s"] += time.perf_counter() - t
            resp, stop = r["tokens"], r["stop"]
        allowed = self.quota[cat] - self.scored[cat]
        n = min(len(resp), allowed)
        term = "quota" if n < len(resp) else stop
        t = time.perf_counter()
        tlp = teacher_logprobs(self.teacher, prompt, resp[:n], self.dev)
        self._sync()
        self.time["teacher_scoring_s"] += time.perf_counter() - t
        self.scored[cat] += n
        rec = {"k": k, "source_id": pr["id"], "category": cat, "rollout_seed": rseed, "response_token_ids": resp,
               "response_len": len(resp), "termination": term, "scored_tokens": n}
        self.presentations.append(rec)
        return {"prompt": prompt, "resp": resp[:n], "tlp": tlp, "n": n, "rec": rec}

    def train(self):
        total = sum(self.quota.values())
        done_tokens = 0
        pending = None  # (presentation, start offset)
        k = 0
        upd = 0
        self.student.train()
        while done_tokens < total:
            N = min(self.tpu, total - done_tokens)  # this update's supervised-token count (normaliser)
            filled = 0
            loss_sum = 0.0
            t_fb = 0.0
            self.opt.zero_grad(set_to_none=True)
            chunks = []
            while filled < N:
                if pending is None:
                    pres = self._next_presentation(k)
                    k += 1
                    if pres is None:
                        break
                    if pres["n"] == 0:
                        continue
                    pending = (pres, 0)
                pres, lo = pending
                hi = min(pres["n"], lo + (N - filled))
                t = time.perf_counter()
                kl = student_kl_sum(self.student, pres["prompt"], pres["resp"], lo, hi, pres["tlp"], self.dev)
                self.scaler.scale(kl / N).backward()
                self._sync()
                t_fb += time.perf_counter() - t
                loss_sum += float(kl)
                chunks.append({"k": pres["rec"]["k"], "lo": lo, "hi": hi})
                filled += hi - lo
                pending = (pres, hi) if hi < pres["n"] else None
            if filled == 0:
                break
            t = time.perf_counter()
            self.scaler.unscale_(self.opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_(self.params, 1.0))
            scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
            self.scaler.step(self.opt)
            self.scaler.update()
            scale_after = self.scaler.get_scale() if self.scaler.is_enabled() else 1.0
            self._sync()
            t_fb += time.perf_counter() - t
            self.time["fwd_bwd_opt_s"] += t_fb
            skipped = scale_after < scale_before
            done_tokens += filled
            upd += 1
            self.updates.append({"update": upd, "tokens": filled, "normaliser": N, "loss_mean_kl": loss_sum / filled,
                                 "lr": self.opt.param_groups[0]["lr"], "grad_norm_unscaled": gnorm,
                                 "scale": scale_before, "skipped_overflow": skipped,
                                 "cum_scored": dict(self.scored), "chunks": chunks})
            if upd % 10 == 0 or done_tokens >= total:
                self.log(f"[{self.method} s{self.seed}] update {upd} tokens {done_tokens}/{total} "
                         f"kl {loss_sum / filled:.4f} gnorm {gnorm:.3f} scale {scale_before:g} skipped {skipped}")
        return {"updates": len(self.updates), "skipped_updates": sum(u["skipped_overflow"] for u in self.updates),
                "scored": dict(self.scored), "presentations": len(self.presentations), "time": dict(self.time)}


@torch.no_grad()
def teacher_collection(teacher, tok, prompts: dict[str, list[dict]], device, batch: int = 16) -> dict:
    """One temperature-1 teacher response per training prompt (caps 192 / 32); per-prompt seed recorded."""
    out = {}
    for cat, ps in prompts.items():
        for b in range(0, len(ps), batch):
            chunk = ps[b:b + batch]
            seeds = [7_000_000 + (0 if cat == "math" else 100_000) + b + i for i in range(len(chunk))]
            res = batched_sample(teacher, [chat_ids(tok, p["text"]) for p in chunk], [CAPS_TRAIN[cat]] * len(chunk),
                                 seeds, EOS_IDS, device, pad_id=tok.pad_token_id)
            for p, s, r in zip(chunk, seeds, res):
                out[p["id"]] = {"tokens": r["tokens"], "stop": r["stop"], "seed": s, "category": cat}
    return out


@torch.no_grad()
def merge_adapter(fresh_backbone, adapter_dir):
    from peft import PeftModel

    pm = PeftModel.from_pretrained(fresh_backbone, adapter_dir)
    return pm.merge_and_unload()
