# Decisions

Selection rules are recorded here BEFORE any held-out test evaluation runs (CLAUDE.md).
Each entry: date (UTC), task/step, decision, selection data used (calibration/dev only), rationale, commit.

| date | task/step | decision | selection data | rationale | commit |
|---|---|---|---|---|---|
| 2026-11-25 | Task 3 / prefix banks | Build teacher + original-draft banks with `scripts/build_prefix_banks.py`: course `setup_data.py` logic executed unchanged except the `generate()` call (see deviations D1). Banks are a local deterministic build with a sha256 manifest. | n/a (data build) | course script crashes under pinned transformers 4.57.1 | (this commit) |
| 2026-11-25 | Task 3 / speculative bank (iii) | We generate the exact-speculative bank ourselves with our Task 3 exact sampler (original student, gamma=4), as the course data README instructs (see D2). | n/a | bank not shipped by the course repo | (this commit) |
| 2026-11-25 | Task 3 / short-form held-out templates | Held-out template families = staff `split=shifted` test prompts (30); familiar = `split=familiar` (30). No additional reservation from the 300 training prompts (they use exactly 3 templates, one per `family` label 0/1/2). | staff split labels | staff split already reserves unseen templates | (this commit) |
| 2026-11-25 | Task 1 / WikiText-2 windows | The staff window manifest (`wikitext2_test_windows_locked`, one document per HF row, 437 windows, 50,000 scored tokens) is authoritative. | n/a | course data repo is authoritative for selections | (this commit) |
