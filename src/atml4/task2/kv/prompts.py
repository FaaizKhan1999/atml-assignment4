"""Task 2C retrieval prompts: loading, chat-template rendering, context/question split, evidence mapping, extractor.

Split rule (docs/decisions.md): every user_text ends with "\\nQUESTION_BEGIN\\n<question>", where <question> is
"What is the access code for CASE-...? Reply with only five digits." (asserted for all 180 prompts). The rendered
prompt is tokenizer.apply_chat_template([{"role": "user", "content": user_text}], add_generation_prompt=True,
tokenize=False) (default Qwen system prompt included). CONTEXT = all tokens before the first token of the question
text (so "QUESTION_BEGIN\\n" belongs to the context); SUFFIX = question tokens + "<|im_end|>\\n" + assistant header.
The split is defined on token IDs of the full sequence (first token whose start offset >= question start); it is
asserted that no token straddles the boundary and that tokenizing both halves separately gives the same IDs.
Gold evidence offsets are mapped to token indices for EVALUATION ONLY.
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field

from atml4.common.meta import repo_root

PATH = repo_root() / "external/atml-pa4-datasets/data/kv_retrieval/kv_retrieval_cases.jsonl.gz"
MARKER = "\nQUESTION_BEGIN\n"
QUESTION_RE = re.compile(r"\nQUESTION_BEGIN\n(What is the access code for CASE-[0-9A-Z-]+\? Reply with only five digits\.)\Z")
CODE_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")
MAX_NEW_TOKENS = 16


def load_cases() -> list[dict]:
    with gzip.open(PATH, "rt") as f:
        return [json.loads(line) for line in f]


def extract_answer(text: str) -> str | None:
    """First maximal run of EXACTLY five consecutive digits in the decoded output; None = 'no answer'."""
    m = CODE_RE.search(text)
    return m.group(0) if m else None


@dataclass
class Prompt:
    id: str
    case: dict
    ids: list[int]              # full sequence (context + suffix)
    L: int                      # context length (tokens before the question)
    suffix_ids: list[int]
    evidence_tokens: list[int]  # token indices overlapping the gold evidence span (evaluation only)
    answer_ids: list[int]       # gold code tokens + <|im_end|>
    rendered: str = field(repr=False, default="")

    @property
    def context_ids(self) -> list[int]:
        return self.ids[:self.L]


def build(case: dict, tok) -> Prompt:
    ut = case["user_text"]
    m = QUESTION_RE.search(ut)
    if m is None or ut.count("QUESTION_BEGIN") != 1:
        raise ValueError(f"{case['id']}: no clean question marker")
    s = tok.apply_chat_template([{"role": "user", "content": ut}], add_generation_prompt=True, tokenize=False)
    u0 = s.index(ut)
    q0 = u0 + m.start(1)
    enc = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    L = next(i for i, (a, _) in enumerate(off) if a >= q0)
    if off[L - 1][1] > q0:
        raise AssertionError(f"{case['id']}: token straddles the question boundary")
    c = tok(s[:q0], add_special_tokens=False)["input_ids"]
    d = tok(s[q0:], add_special_tokens=False)["input_ids"]
    if c + d != ids or len(c) != L:
        raise AssertionError(f"{case['id']}: separate tokenization differs at the boundary")
    es, ee = u0 + case["evidence_char_start"], u0 + case["evidence_char_end"]
    assert s[es:ee] == case["evidence_text"]
    ev = [i for i, (a, b) in enumerate(off) if a < ee and b > es]
    assert ev and max(ev) < L, "evidence must lie in the context"
    ans = tok(case["code"], add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
    return Prompt(case["id"], case, ids, L, ids[L:], ev, ans, s)


def cell(case: dict) -> tuple[int, str, int]:
    return case["target_tokens"], case["position"], case["distractors"]
