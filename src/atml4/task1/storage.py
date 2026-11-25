"""Storage accounting for Task 1 (pure functions; LOGICAL/serialized bytes, never GPU-allocated bytes).

Conventions
-----------
* Dense FP16: 2 bytes per element.
* Model total = eligible-weight bytes (in the chosen format) + every other parameter at its dense dtype size
  (embeddings, norms, biases, lm_head), with shared/tied tensors counted once.
* Unstructured 16-bit sparse format (PDF Task 1 "Shared resources"): per eligible matrix, a bitmask with one bit
  per weight plus 16 bits per nonzero value. Padding: the bitmask of EACH matrix is rounded up to whole bytes,
  i.e. mask_bytes = ceil(numel / 8); values are 2 bytes each so need no padding. No index/offset arrays.
* Group-wise asymmetric b-bit quantized format (group size along input columns): see quantized_matrix_bytes.
  Defaults follow the fixed convention in docs/decisions.md (group 128, FP16 scale, b-bit zero-point, one
  little-endian bitstream per matrix padded to a whole byte) and equal the real packer's output length.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

FP16_BYTES = 2


def dense_bytes(numel: int, bytes_per_el: int = FP16_BYTES) -> int:
    return int(numel) * int(bytes_per_el)


def sparse16_matrix_bytes(numel: int, nnz: int) -> dict:
    """Bitmask (1 bit / weight, padded per matrix to whole bytes) + 16 bits per nonzero."""
    if not 0 <= nnz <= numel:
        raise ValueError("nnz must be in [0, numel]")
    mask = math.ceil(numel / 8)
    values = 2 * nnz
    return {"mask_bytes": mask, "value_bytes": values, "padding_bits": mask * 8 - numel,
            "total_bytes": mask + values}


def quantized_matrix_bytes(d_out: int, d_in: int, *, bits: int, group_size: int = 128, scale_bits: int = 16,
                           zero_bits: int | None = None, pad_columns_to_group: bool = True,
                           joint_bitstream: bool = True, payload_align_bits: int = 8,
                           metadata_align_bits: int = 8) -> dict:
    """Packed bytes of one d_out x d_in matrix quantized group-wise along input columns.

    Defaults = fixed Task 1 convention (docs/decisions.md): group 128, FP16 scale (16 bits), b-bit zero-point
    (zero_bits=None -> bits), padded columns stored, ONE bitstream per matrix (codes + zero-points + scales)
    padded to a whole byte (joint_bitstream=True, payload_align_bits=8). With joint_bitstream=False payload and
    metadata are aligned separately to payload_align_bits / metadata_align_bits.
    groups per row = ceil(d_in / group_size).
    """
    zero_bits = bits if zero_bits is None else zero_bits
    for name, v in [("bits", bits), ("group_size", group_size), ("payload_align_bits", payload_align_bits),
                    ("metadata_align_bits", metadata_align_bits)]:
        if v <= 0:
            raise ValueError(f"{name} must be positive")
    if scale_bits < 0 or zero_bits < 0:
        raise ValueError("metadata bit-widths must be >= 0")
    groups_per_row = math.ceil(d_in / group_size)
    cols = groups_per_row * group_size if pad_columns_to_group else d_in
    payload_bits = d_out * cols * bits
    n_groups = d_out * groups_per_row
    meta_bits = n_groups * (scale_bits + zero_bits)
    if joint_bitstream:
        total_bits = math.ceil((payload_bits + meta_bits) / payload_align_bits) * payload_align_bits
        if total_bits % 8:
            raise ValueError("alignment must yield whole bytes")
        total = total_bits // 8
        payload, meta = payload_bits / 8, meta_bits / 8  # logical split (not separately byte-aligned)
        pad_bits = total_bits - payload_bits - meta_bits
    else:
        pa = math.ceil(payload_bits / payload_align_bits) * payload_align_bits
        ma = math.ceil(meta_bits / metadata_align_bits) * metadata_align_bits if meta_bits else 0
        if pa % 8 or ma % 8:
            raise ValueError("alignment must yield whole bytes (use multiples of 8 bits)")
        payload, meta, total = pa // 8, ma // 8, (pa + ma) // 8
        pad_bits = (pa - payload_bits) + (ma - meta_bits)
    return {"payload_bytes": payload, "metadata_bytes": meta, "total_bytes": total, "n_groups": n_groups,
            "padding_bits": pad_bits + (cols - d_in) * d_out * bits, "nominal_bits": bits,
            "effective_bits_per_weight": 8 * total / (d_out * d_in)}


def two_four_matrix_bytes(numel: int) -> dict:
    """2:4 representation ESTIMATE: FP16 kept values (numel/2) + 2-bit position index per kept value,
    padded per matrix to whole bytes. No sparse-tensor-core execution implied."""
    if numel % 4:
        raise ValueError("numel must be divisible by 4")
    kept = numel // 2
    bits = kept * (16 + 2)
    return {"kept": kept, "value_bytes": 2 * kept, "index_bytes": math.ceil(kept * 2 / 8),
            "total_bytes": math.ceil(bits / 8)}


@dataclass
class ModelStorage:
    eligible_params: int
    eligible_fp16_bytes: int
    other_params: int
    other_bytes: int
    total_dense_fp16_bytes: int

    def to_dict(self):
        return asdict(self)


def model_storage(eligible_numels: list[int], other_param_sizes: list[tuple[int, int]]) -> ModelStorage:
    """eligible_numels: numel per eligible matrix; other_param_sizes: (numel, bytes_per_el) per UNIQUE other
    parameter (tied tensors already deduplicated by the caller)."""
    e = sum(eligible_numels)
    o = sum(n for n, _ in other_param_sizes)
    ob = sum(n * b for n, b in other_param_sizes)
    return ModelStorage(e, dense_bytes(e), o, ob, dense_bytes(e) + ob)


def sparse16_model_bytes(matrices: list[tuple[int, int]], other_bytes: int) -> dict:
    """matrices: (numel, nnz) per eligible matrix. Returns sparse-format total next to dense FP16."""
    per = [sparse16_matrix_bytes(n, z) for n, z in matrices]
    elig = sum(p["total_bytes"] for p in per)
    numel = sum(n for n, _ in matrices)
    nnz = sum(z for _, z in matrices)
    return {"format": "unstructured16: 1 mask bit/weight (per-matrix byte padding) + 16 bits/nonzero",
            "eligible_bytes": elig, "mask_bytes": sum(p["mask_bytes"] for p in per),
            "value_bytes": sum(p["value_bytes"] for p in per), "other_bytes": other_bytes,
            "total_bytes": elig + other_bytes, "dense_fp16_eligible_bytes": dense_bytes(numel),
            "dense_fp16_total_bytes": dense_bytes(numel) + other_bytes,
            "nonzero_fraction": nnz / numel if numel else float("nan"),
            "effective_bits_per_eligible_weight": 8 * elig / numel if numel else float("nan")}


def model_storage_from_torch(model) -> tuple[ModelStorage, list[dict]]:
    """Measure storage of a loaded model; also returns per-eligible-matrix (numel, nnz)."""
    from .model import eligible_projections, eligible_weight_names, unique_parameters

    elig_names = eligible_weight_names(model)
    per = []
    for p in eligible_projections(model):
        w = p.module.weight
        per.append({"qualname": p.qualname, "layer": p.layer, "name": p.name, "shape": list(p.shape),
                    "numel": w.numel(), "nnz": int((w != 0).sum().item())})
    others = [(t.numel(), t.element_size()) for n, t in unique_parameters(model) if n not in elig_names]
    return model_storage([x["numel"] for x in per], others), per
