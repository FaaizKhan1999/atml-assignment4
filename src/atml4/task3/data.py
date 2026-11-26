"""Task 3 data: held-out prompts (120), training prompts (800), chat formatting, prefix banks, answer scoring.

Prompt format (identical for every model/condition, = course prefix-bank code): tokenizer.apply_chat_template(
[{"role": "user", "content": text}], add_generation_prompt=True) with no system message (Qwen's default system
prompt is inserted by the template); non-thinking. Target and draft tokenizers/templates are identical (checked).
Prompt IDs follow the course bank code: GSM8K-{order:04d} (manifest order) and the short-form `id` field.
"""
from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path

from atml4.common.meta import repo_root

DATA = repo_root() / "external/atml-pa4-datasets/data"
BANK_PATHS = [DATA / "prefix_banks/prefix_banks_locked.jsonl.gz",
              repo_root() / "kaggle_runs/prefix-banks/output/prefix_banks/prefix_banks_locked.jsonl.gz"]
EOS_IDS = (151645, 151643)  # Instruct generation_config eos_token_id: <|im_end|>, <|endoftext|> (same for both models)
CAPS_SPEED = 64
CAPS_QUALITY = {"math": 512, "short": 32}
CAPS_TRAIN = {"math": 192, "short": 32}
BANK_BINS = [(1, 16, 8), (17, 32, 8), (33, 64, 16), (65, 192, 16)]  # course prefix_banks(): (lo, hi, n)


def _jsonl(path: Path) -> list[dict]:
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_prompts() -> list[dict]:
    """120 held-out prompts in the course bank order: 60 GSM8K test (manifest order) then 60 short-form test."""
    out = []
    for r in _jsonl(DATA / "public_locked/gsm8k_test_locked.jsonl.gz"):
        ans = r["example"]["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"GSM8K-{r['order']:04d}", "text": r["example"]["question"], "domain": "math",
                    "workload": "math", "answer": ans})
    for r in _jsonl(DATA / "short_form/short_test.jsonl"):
        out.append({"id": r["id"], "text": r["input"], "domain": "short", "workload": r["split"],
                    "answer": r["answer"], "family": r["family"]})
    assert len(out) == 120
    return out


def train_prompts() -> dict[str, list[dict]]:
    math = [{"id": f"GSM8K-train-{r['order']:04d}", "text": r["example"]["question"], "category": "math"}
            for r in _jsonl(DATA / "public_locked/gsm8k_train_locked.jsonl.gz")]
    short = [{"id": r["id"], "text": r["input"], "category": "short"} for r in _jsonl(DATA / "short_form/short_train.jsonl")]
    assert len(math) == 500 and len(short) == 300
    return {"math": math, "short": short}


def chat_ids(tok, text: str) -> list[int]:
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)


def bank_seed(j: int, bank: str) -> int:
    """Course per-sequence seed: 6304 + 10007 j (+123 for the original draft). Our speculative and own-student
    banks use the teacher offset (0)."""
    return 6304 + 10007 * j + (123 if bank == "original_draft" else 0)


def bank_positions(pid: str, which: str, seq_len: int) -> list[int]:
    """Course position-selection logic (setup_data.prefix_banks), reused verbatim."""
    rng = random.Random(f"PA4-banks-v1:6304:{pid}:{which}")
    positions = []
    for lo, hi, n in BANK_BINS:
        candidates = list(range(lo, min(hi, seq_len) + 1))
        positions.extend(sorted(rng.sample(candidates, min(n, len(candidates)))))
    return positions


def bank_records(pid, domain, which, prompt, seq, seed, extra=None) -> list[dict]:
    return [dict(prompt_id=pid, domain=domain, bank=which, position=pos, prompt=prompt,
                 response_prefix_token_ids=seq[:pos - 1], source_length=len(seq), source_generation_seed=seed,
                 **(extra or {})) for pos in bank_positions(pid, which, len(seq))]


def load_course_banks() -> list[dict]:
    for p in BANK_PATHS:
        if p.exists():
            return _jsonl(p)
    raise FileNotFoundError("prefix_banks_locked.jsonl.gz not found (mount focusedkognition/atml4-prefix-banks)")


# ---------------------------------------------------------------- answer scoring (Step 7, documented rules)
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm8k_extract(text: str) -> str | None:
    """Rule: if '\\boxed{...}' occurs, the last number inside the last \\boxed{}; else if '####' occurs, the first
    number after the last '####'; else the LAST number in the text. Commas removed; trailing '.0' normalised."""
    seg = None
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        seg = boxed[-1]
        nums = NUM_RE.findall(seg)
        return _norm(nums[-1]) if nums else None
    if "####" in text:
        nums = NUM_RE.findall(text.split("####")[-1])
        return _norm(nums[0]) if nums else None
    nums = NUM_RE.findall(text)
    return _norm(nums[-1]) if nums else None


def _norm(s: str) -> str:
    s = s.replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return s
    return str(int(v)) if v == int(v) else str(v)


def short_score(text: str, answer: str) -> dict:
    """compliant = the stripped output (one trailing '.' removed) equals the answer exactly (format compliance);
    correct = case-insensitive equality after the same normalisation; contains = answer appears as a token."""
    o = text.strip()
    o1 = o[:-1].strip() if o.endswith(".") else o
    return {"compliant": o1 == answer, "correct": o1.lower() == answer.lower(),
            "contains": re.search(rf"(?<![\w]){re.escape(answer)}(?![\w])", o, re.I) is not None}
