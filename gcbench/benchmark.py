from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .metrics import off_diagonal_mask, summarize_edge_recovery
from .models import create_model
from .simulations import PAPER_SYSTEMS, generate_paper_system
from .stats import bh_fdr, block_permute, permutation_p_value


@dataclass
class BenchmarkResult:
    model_name: str
    max_lag: int
    train_fraction: float
    val_fraction: float
    n_permutations: int
    block_size: int
    q_value: float
    runtime_seconds: float
    score_matrix: np.ndarray
    pvalue_matrix: np.ndarray
    significant_matrix: np.ndarray
    unrestricted_mse: np.ndarray
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "max_lag": self.max_lag,
            "train_fraction": self.train_fraction,
            "val_fraction": self.val_fraction,
            "n_permutations": self.n_permutations,
            "block_size": self.block_size,
            "q_value": self.q_value,
            "runtime_seconds": self.runtime_seconds,
            "score_matrix": self.score_matrix.tolist(),
            "pvalue_matrix": self.pvalue_matrix.tolist(),
            "significant_matrix": self.significant_matrix.astype(int).tolist(),
            "unrestricted_mse": self.unrestricted_mse.tolist(),
            "metrics": self.metrics,
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def split_time_series(
    data: np.ndarray,
    *,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data.ndim != 2:
        raise ValueError("Expected data with shape (time, variables).")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    n_time = data.shape[0]
    train_end = int(n_time * train_fraction)
    val_end = int(n_time * (train_fraction + val_fraction))
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    if min(train.shape[0], test.shape[0]) < 2:
        raise ValueError("Split produced an empty train or test segment.")
    return train, val, test


def run_granger_inference(
    data: np.ndarray,
    *,
    model_name: str = "var",
    max_lag: int = 10,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    n_permutations: int = 100,
    block_size: int | None = None,
    q_value: float = 0.05,
    random_state: int | None = None,
    model_kwargs: dict[str, object] | None = None,
) -> BenchmarkResult:
    train, val, test = split_time_series(data, train_fraction=train_fraction, val_fraction=val_fraction)
    fit_data = np.concatenate([train, val], axis=0)
    if fit_data.shape[0] <= max_lag:
        raise ValueError("Not enough train+validation data for the requested lag order.")
    if test.shape[0] <= 2:
        raise ValueError("Need at least 3 test samples for held-out evaluation.")

    n_vars = data.shape[1]
    scores = np.zeros((n_vars, n_vars), dtype=float)
    pvalues = np.ones((n_vars, n_vars), dtype=float)
    unrestricted_mse = np.zeros(n_vars, dtype=float)
    rng = np.random.default_rng(random_state)
    block_size = block_size or max(max_lag, 4)
    model_kwargs = model_kwargs or {}
    start_time = perf_counter()

    for target in range(n_vars):
        target_seed = None if random_state is None else random_state + target * 1000
        unrestricted = create_model(model_name, max_lag, random_state=target_seed, **model_kwargs)
        unrestricted.fit(train, target, included_sources=tuple(range(n_vars)), val_data=val)
        unrestricted_mse[target] = unrestricted.mse_segment(fit_data, test)

        for source in range(n_vars):
            if source == target:
                continue

            restricted_sources = tuple(idx for idx in range(n_vars) if idx != source)
            edge_seed = None if target_seed is None else target_seed + source + 1
            restricted = create_model(model_name, max_lag, random_state=edge_seed, **model_kwargs)
            restricted.fit(train, target, included_sources=restricted_sources, val_data=val)
            restricted_mse = restricted.mse_segment(fit_data, test)
            actual_score = restricted_mse - unrestricted_mse[target]
            scores[source, target] = actual_score

            permuted_scores = np.empty(n_permutations, dtype=float)
            for perm_idx in range(n_permutations):
                permuted_test = np.array(test, copy=True)
                permuted_test[:, source] = block_permute(permuted_test[:, source], block_size, rng)
                permuted_mse = unrestricted.mse_segment(fit_data, permuted_test)
                permuted_scores[perm_idx] = restricted_mse - permuted_mse

            pvalues[source, target] = permutation_p_value(actual_score, permuted_scores)

    runtime_seconds = perf_counter() - start_time
    mask = off_diagonal_mask(n_vars)
    rejected, _ = bh_fdr(pvalues, q=q_value, mask=mask)
    significant = rejected & (scores > 0.0) & mask
    metrics = summarize_edge_recovery(np.zeros_like(significant), scores, significant, mask=mask)

    return BenchmarkResult(
        model_name=model_name,
        max_lag=max_lag,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        n_permutations=n_permutations,
        block_size=block_size,
        q_value=q_value,
        runtime_seconds=runtime_seconds,
        score_matrix=scores,
        pvalue_matrix=pvalues,
        significant_matrix=significant,
        unrestricted_mse=unrestricted_mse,
        metrics=metrics,
    )


def run_paper_system_benchmark(
    system_id: str,
    *,
    model_name: str = "var",
    length: int = 1200,
    max_lag: int = 10,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
    n_permutations: int = 100,
    block_size: int | None = None,
    q_value: float = 0.05,
    random_state: int | None = None,
    model_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, BenchmarkResult]:
    if system_id not in PAPER_SYSTEMS:
        valid = ", ".join(sorted(PAPER_SYSTEMS))
        raise ValueError(f"Unknown paper system '{system_id}'. Expected one of: {valid}")

    data, truth = generate_paper_system(system_id, length=length, seed=random_state)
    result = run_granger_inference(
        data,
        model_name=model_name,
        max_lag=max_lag,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        n_permutations=n_permutations,
        block_size=block_size,
        q_value=q_value,
        random_state=random_state,
        model_kwargs=model_kwargs,
    )
    mask = off_diagonal_mask(truth.shape[0])
    result.metrics = summarize_edge_recovery(truth, result.score_matrix, result.significant_matrix, mask=mask)
    return data, truth, result
