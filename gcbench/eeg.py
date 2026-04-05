from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConditionNetworks:
    condition: str
    subject_ids: tuple[str, ...]
    matrices: np.ndarray


def load_preprocessed_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def validate_epoch_array(epochs: np.ndarray) -> None:
    if epochs.ndim != 3:
        raise ValueError("Expected epochs with shape (n_epochs, n_times, n_channels).")
    if epochs.shape[0] == 0 or epochs.shape[1] == 0 or epochs.shape[2] == 0:
        raise ValueError("Epoch array must be non-empty in every dimension.")


def concatenate_epochs(epochs: np.ndarray) -> np.ndarray:
    validate_epoch_array(epochs)
    n_epochs, n_times, n_channels = epochs.shape
    return epochs.reshape(n_epochs * n_times, n_channels)


def mean_network(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3:
        raise ValueError("Expected matrices with shape (subjects, nodes, nodes).")
    return matrices.mean(axis=0)


def edge_frequency(matrices: np.ndarray, *, threshold: float = 0.0) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=float)
    if matrices.ndim != 3:
        raise ValueError("Expected matrices with shape (subjects, nodes, nodes).")
    return (matrices > threshold).mean(axis=0)


def graph_summary(matrix: np.ndarray, *, threshold: float = 0.0) -> dict[str, np.ndarray | float]:
    matrix = np.asarray(matrix, dtype=float)
    binary = matrix > threshold
    np.fill_diagonal(binary, False)
    density = float(binary.sum() / max(binary.size - matrix.shape[0], 1))
    in_degree = binary.sum(axis=0)
    out_degree = binary.sum(axis=1)
    return {
        "density": density,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "mean_edge_weight": float(matrix[binary].mean()) if np.any(binary) else 0.0,
    }


def condition_difference(
    condition_a: np.ndarray,
    condition_b: np.ndarray,
) -> np.ndarray:
    condition_a = np.asarray(condition_a, dtype=float)
    condition_b = np.asarray(condition_b, dtype=float)
    if condition_a.shape != condition_b.shape:
        raise ValueError("Condition arrays must have matching shapes.")
    return condition_a.mean(axis=0) - condition_b.mean(axis=0)


def jaccard_stability(binary_matrices: np.ndarray) -> float:
    binary_matrices = np.asarray(binary_matrices, dtype=bool)
    if binary_matrices.ndim != 3:
        raise ValueError("Expected matrices with shape (subjects, nodes, nodes).")
    if binary_matrices.shape[0] < 2:
        return 1.0

    scores: list[float] = []
    for idx in range(binary_matrices.shape[0]):
        for jdx in range(idx + 1, binary_matrices.shape[0]):
            left = binary_matrices[idx]
            right = binary_matrices[jdx]
            union = np.count_nonzero(left | right)
            if union == 0:
                scores.append(1.0)
                continue
            intersection = np.count_nonzero(left & right)
            scores.append(intersection / union)
    return float(np.mean(scores))


def condition_table(networks_by_condition: Mapping[str, np.ndarray]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for condition, matrices in networks_by_condition.items():
        summary = graph_summary(mean_network(matrices))
        rows.append(
            {
                "condition": condition,
                "density": float(summary["density"]),
                "mean_edge_weight": float(summary["mean_edge_weight"]),
                "jaccard_stability": jaccard_stability(matrices > 0.0),
            }
        )
    return rows
