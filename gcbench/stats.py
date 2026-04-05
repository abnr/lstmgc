from __future__ import annotations

import numpy as np


def block_permute(values: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("block_permute expects a one-dimensional array.")
    if values.size == 0:
        return values.copy()

    block_size = max(1, min(block_size, values.size))
    blocks = [values[start : start + block_size] for start in range(0, values.size, block_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[idx] for idx in order])[: values.size]


def permutation_p_value(actual_score: float, permuted_scores: np.ndarray) -> float:
    permuted_scores = np.asarray(permuted_scores, dtype=float)
    if permuted_scores.size == 0:
        return 1.0
    extreme = np.count_nonzero(permuted_scores >= actual_score)
    return float((extreme + 1) / (permuted_scores.size + 1))


def bh_fdr(
    p_values: np.ndarray,
    *,
    q: float = 0.05,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    p_values = np.asarray(p_values, dtype=float)
    if mask is None:
        mask = np.ones_like(p_values, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    flat = p_values[mask]
    valid = np.isfinite(flat)
    flat = flat[valid]
    if flat.size == 0:
        return np.zeros_like(p_values, dtype=bool), float("nan")

    order = np.argsort(flat)
    sorted_p = flat[order]
    thresholds = q * np.arange(1, sorted_p.size + 1) / sorted_p.size
    passed = sorted_p <= thresholds

    if not np.any(passed):
        return np.zeros_like(p_values, dtype=bool), float("nan")

    cutoff = float(sorted_p[np.where(passed)[0].max()])
    rejected = np.zeros_like(p_values, dtype=bool)
    rejected[mask] = p_values[mask] <= cutoff
    return rejected, cutoff
