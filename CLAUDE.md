# CLAUDE.md — standing rules for ATML PA4 (EE-5102/CS-6304, due 30 Nov 2026)

Role: implementation engineer. A separate reviewer writes instructions; the user relays them.

## Ground rules
- The assignment PDF is `private/assignment.pdf` (gitignored). Read it before any task work. If missing, STOP and ask.
- Follow the assignment to the letter. Never invent a dataset path, manifest field, hyperparameter, model revision,
  API behaviour or result. If something is ambiguous or missing, stop and ask in the report instead of guessing.
- Generative AI writes CODE only. Never write report prose; never create report `.tex` body text.
- Git: commit early and often with conventional messages (`feat:`, `fix:`, `exp:`, `data:`, `docs:`, `chore:`,
  `test:`) and push after every logical unit.
  - NO Claude/AI attribution anywhere: no `Co-Authored-By: Claude`, no "Generated with Claude Code"
    (disabled in `.claude/settings.json`; verify with `git log --format=%B -n 3`).
  - Git author: Faaiz Khan (global git identity).

## Experimental rules
- Seed 6304 is the default reference seed. Extra seeds only where the assignment specifies them
  (Task 2B: 6304/6305/6306; Task 3 training: 6304/6305).
- Every result file carries a `meta` block (`atml4.common.meta.collect_meta`) and is reproducible from one command,
  which is logged inside the file (`meta.command`). Use `atml4.common.io.save_result`.
- Every model/tokenizer load passes `revision=` from `configs/models.yaml`.
- Never use held-out test outcomes (WikiText-2 test, C4 val, HellaSwag, MQAR final-test, the 180 KV prompts,
  the 120 Task-3 test prompts) to choose anything. Selection uses calibration/dev subsets only, and the selection
  rule is written to `docs/decisions.md` BEFORE the test evaluation runs.
- Report every specified condition, including failures and OOM runs. Log deviations in `docs/deviations.md`.
- Timing protocol: warm up; `torch.cuda.synchronize()` around timed regions; >=5 reps (>=10 where the PDF says so);
  store all raw samples plus unit and rep count; keep calibration / training / prefill / decode / postprocessing
  separate. Use `atml4.common.timing`.
- Keep these pairs distinct in names and files:
  - logical or packed bytes  vs  GPU-allocated bytes
  - theoretical arithmetic reduction  vs  measured speedup
  - fake-quantized FP16  vs  real low-bit execution
- Shared-machine timing rule: when a kernel uses T4 x2 to run two jobs concurrently (cuda:0 / cuda:1), NO
  latency/throughput measurement may run while the other GPU is busy. Timing runs get the machine to themselves,
  and their result meta records this (`timing_exclusive: true` plus the other-GPU state).
- GPU work runs only on Kaggle via `scripts/kaggle_run.py`. Log every run in `docs/gpu_ledger.md`
  (kernel, accelerator, wall time, purpose). Budget ~30 GPU-hours total: flag any job expected to take > 2 h
  before launching it.
- Never commit raw datasets or full checkpoints (LoRA adapters of a few MB are fine).

## Layout
- `src/atml4/common/` shared utilities; `src/atml4/task{1,2,3}/` task code; `scripts/` CLI entry points
- `configs/` model revisions + experiment YAMLs; `results/task*/` small json/jsonl/csv only
- `kaggle/` kernel templates; `kaggle_runs/` (gitignored) downloaded raw outputs
- `docs/` gpu_ledger.md, decisions.md, deviations.md; `external/atml-pa4-datasets` course data submodule (pinned)

## End-of-session REPORT format
1. Repo URL + latest commit SHA; `git log --format='%h %an %s' -n 10`; confirmation of no attribution trailers
2. Repo tree (depth 3)
3. Datasets-repo / data notes relevant to the session
4. Test results (counts, failures)
5. Kaggle: kernels, statuses, accelerators, wall times; gpu_ledger running total
6. Environment findings
7. Runtime estimates for pending jobs
8. Open questions / anything not done. Do not guess.
