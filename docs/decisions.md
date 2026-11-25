# Decisions

Selection rules are recorded here BEFORE any held-out test evaluation runs (CLAUDE.md).
Each entry: date (UTC), task/step, decision, selection data used (calibration/dev only), rationale, commit.

| date | task/step | decision | selection data | rationale | commit |
|---|---|---|---|---|---|
| 2026-11-25 | Task 3 / prefix banks | Build teacher + original-draft banks with `scripts/build_prefix_banks.py`: course `setup_data.py` logic executed unchanged except the `generate()` call (see deviations D1). Banks are a local deterministic build with a sha256 manifest. | n/a (data build) | course script crashes under pinned transformers 4.57.1 | (this commit) |
| 2026-11-25 | Task 3 / speculative bank (iii) | We generate the exact-speculative bank ourselves with our Task 3 exact sampler (original student, gamma=4), as the course data README instructs (see D2). | n/a | bank not shipped by the course repo | (this commit) |
| 2026-11-25 | Task 3 / short-form held-out templates | Held-out template families = staff `split=shifted` test prompts (30); familiar = `split=familiar` (30). No additional reservation from the 300 training prompts (they use exactly 3 templates, one per `family` label 0/1/2). | staff split labels | staff split already reserves unseen templates | (this commit) |
| 2026-11-25 | Task 1 / WikiText-2 windows | The staff window manifest (`wikitext2_test_windows_locked`, one document per HF row, 437 windows, 50,000 scored tokens) is authoritative. | n/a | course data repo is authoritative for selections | (this commit) |
| 2026-11-25 | Task 1 / HellaSwag diagnostic subset | The first 512 of the 2,000 supplied IDs are 100% ActivityNet (manifest in row-index order; rows 1-2490). Kept exactly as specified; noted for interpretation. | staff manifest | PDF/README define the subset as the first 512 IDs | (this commit) |

## Task 1 fixed conventions (P1, P2, Q1, Q2, 2:4) — recorded 2026-11-25 BEFORE any of these conditions ran

### Pruning (P1 magnitude, P2 Wanda)
- Per output row of every eligible matrix zero exactly k = s * d_in weights (s in {0.25, 0.50, 0.75}; d_in in {1536, 8960} so k is an exact integer; asserted).
- Lowest scores pruned; ties broken by column index (stable ascending sort: among equal scores the lower column index is pruned first).
- Scores: magnitude S = |W_ij|; Wanda S = |W_ij| * ||X_j||_2 with ||X_j||_2 = sqrt(sumsq_j) from `results/task1/baseline/calib_input_stats.npz` (all 128 x 512 calibration tokens, FP32 accumulation).
- Pristine FP16 weights restored (and verified bit-exact) before every condition.
- Reported separately: mask-pruned fraction (by construction) and actual zero fraction (baseline already has 2,914 exact zeros).
- 2:4 diagnostic: Wanda scores; each row split into groups of 4 contiguous input columns; keep the 2 highest, zero the 2 lowest (ties by column index); compared with unstructured 50% Wanda; equal zero counts from the masks asserted. Storage: bitmask-sparse16 formula and a 2:4 representation estimate (FP16 kept values + 2-bit position index per kept value). No sparse-tensor-core speedup claimed (T4).

### Quantization (Q1 uniform, Q2 AWQ-style; identical quantizer later for Q3)
- Weight-only, asymmetric, group size 128 of contiguous input columns within each output row. Zero-padding of the last group (padded entries excluded from min/max) is implemented but asserted unused (d_in divisible by 128).
- Per group: w_min, w_max; s = (w_max - w_min)/(2^b - 1) in FP32, then s := max(s, 2^-24) (smallest positive FP16; covers w_max == w_min) and rounded to FP16 (the stored value); every later use of s uses this FP16 value.
- z = clamp(round(-w_min / s), 0, 2^b - 1) (integer zero-point).
- w_hat = s * (clip(round(w/s) + z, 0, 2^b - 1) - z) (the PDF quantizer), computed in FP32 from the FP16 s and cast to FP16.
- Rounding: torch.round (round-half-to-even).
- Serialized per group: b-bit zero-point, 16-bit FP16 scale, 128 b-bit codes => 128b + b + 16 bits per group. Layout: per matrix ONE little-endian bitstream, row-major over (row, group), each group as [zero-point][scale][codes], each field LSB first; the matrix bitstream is padded to a whole byte. A real packer/unpacker is used; serialized length must equal the storage formula and unpack(pack) must reproduce the dequantized FP16 tensor bit-exactly (checked for every matrix in every quantized condition).
- Evaluation uses dequantized FP16 weights ("fake-quantized FP16"); latency is dense FP16 execution, not low-bit execution.

### Q2 AWQ-style scaling
- Scale groups per block, inputs captured from the UNCHANGED FP16 model one block at a time over all 128 x 512 calibration tokens: (1) q/k/v (input_layernorm output), (2) o_proj (attention output), (3) gate/up (post_attention_layernorm output), (4) down_proj (act(gate)*up).
- Per input channel: s = clamp(mean_t |x_tj|^alpha, min 1e-4), normalised by sqrt(max(s)*min(s)) (llm-awq `auto_scale` formula); scales rounded to FP16.
- Grid: alpha in {0, 0.05, ..., 0.95} (20 values). alpha = 0 gives s = 1, i.e. exactly Q1.
- Objective: MSE of the scale group's projection outputs (all projections of the group concatenated, no bias) between W X and [Q(W diag(s)) diag(1/s)] X on all calibration tokens; argmin over the grid (ties -> smaller alpha). One search per bit width; the captured activations are shared by the three bit widths. Note: llm-awq inspects the self_attn / mlp module outputs for q/k/v and gate/up; we use the projection outputs as specified by the reviewer.
- Inverse-scale compensation: down_proj scales are folded into up_proj's output rows (W_up <- diag(1/s) W_up) BEFORE up_proj is scaled/quantized (order: all four scale vectors are searched on the original block weights, then folded/applied, then quantized, as in llm-awq). o_proj <- v_proj folding is invalid under GQA (v_proj out 256 != o_proj in 1536) -> o_proj scales serialized. q/k/v and gate/up follow RMSNorm (not a linear projection) -> FP16 scales serialized (not folded into norm weights). Serialized scale bytes count toward Q2 storage. In the fake-quant evaluation each serialized compensation is merged into the dequantized weight (W_eff = Q(W diag(s)) diag(1/s), FP32 then FP16).
- Verified before rounding: rescaled-but-unquantized model (scaled weights, down->up folding, explicit runtime division of inputs for serialized groups) vs baseline logits on 4 calibration sequences (max abs logit diff reported).
- Reproduced from published AWQ: activation-aware per-channel scale search over a recorded grid, scale folding into the preceding linear layer where valid. NOT reproduced: AWQ weight-clipping search (would change the shared quantizer), fused INT4 kernels / inference engine, AWQ's module-output objective for q/k/v and gate/up.
