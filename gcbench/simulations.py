from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PAPER_SYSTEMS = {
    "system1": "Linear stationary fixed lags",
    "system2": "Linear with varying lag",
    "system3": "Nonlinear polynomial",
    "system4": "Nonlinear polynomial with varying lag",
    "system5": "Nonstationary drift",
    "system6": "Nonstationary drift with varying lag",
    "system7": "Time-varying sine coefficients",
    "system8": "Time-varying cosine coefficients",
}


@dataclass(frozen=True)
class SparseSimulationConfig:
    n_nodes: int = 10
    length: int = 1000
    max_lag: int = 5
    density: float = 0.1
    noise_scale: float = 0.5
    drift_strength: float = 0.0
    time_varying: bool = False
    nonlinear: bool = False
    latent_common_driver: bool = False
    warmup: int = 200
    seed: int | None = None


def paper_truth_adjacency() -> np.ndarray:
    truth = np.zeros((5, 5), dtype=bool)
    truth[0, 1] = True
    truth[0, 2] = True
    truth[0, 3] = True
    truth[3, 4] = True
    truth[4, 3] = True
    return truth


def generate_paper_system(
    system_id: str,
    length: int,
    *,
    warmup: int = 300,
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if system_id not in PAPER_SYSTEMS:
        valid = ", ".join(sorted(PAPER_SYSTEMS))
        raise ValueError(f"Unknown system '{system_id}'. Expected one of: {valid}")

    rng = np.random.default_rng(seed)
    sqrt2 = np.sqrt(2.0)
    max_required_lag = 10
    total = warmup + length + max_required_lag + 1
    x = np.zeros((total, 5), dtype=float)

    for t in range(max_required_lag, total):
        e = rng.normal(0.0, noise_scale, size=5)
        drift = 0.001 * (t - max_required_lag)
        sin_coeff = np.sin(0.001 * (t - max_required_lag) * np.pi)
        cos_coeff = np.cos(0.001 * (t - max_required_lag) * np.pi)

        x[t, 0] = 0.95 * sqrt2 * x[t - 1, 0] - 0.9025 * x[t - 2, 0] + e[0]

        if system_id in {"system1", "system5"}:
            x[t, 1] = 0.5 * x[t - 2, 0] + e[1]
        elif system_id in {"system2", "system6"}:
            x[t, 1] = 0.5 * x[t - 10, 0] + e[1]
        elif system_id == "system3":
            x[t, 1] = 0.5 * x[t - 2, 0] ** 2 + e[1]
        elif system_id == "system4":
            x[t, 1] = 0.5 * x[t - 10, 0] ** 2 + e[1]
        elif system_id == "system7":
            x[t, 1] = 0.5 * sin_coeff * x[t - 2, 0] + e[1]
        elif system_id == "system8":
            x[t, 1] = 0.5 * cos_coeff * x[t - 10, 0] + e[1]

        if system_id in {"system1", "system2", "system5", "system6", "system7", "system8"}:
            x[t, 2] = -0.4 * x[t - 3, 0] + e[2]
        else:
            x[t, 2] = -0.4 * x[t - 3, 0] + e[2]

        if system_id in {"system1", "system2", "system5", "system6"}:
            drive = -0.5 * x[t - 2, 0]
        elif system_id in {"system3", "system4"}:
            drive = -0.5 * x[t - 2, 0] ** 2
        elif system_id == "system7":
            drive = -0.5 * sin_coeff * x[t - 2, 0]
        else:
            drive = -0.5 * cos_coeff * x[t - 2, 0]

        x[t, 3] = drive + 0.25 * sqrt2 * x[t - 1, 3] + 0.25 * sqrt2 * x[t - 1, 4] + e[3]
        x[t, 4] = -0.25 * x[t - 1, 3] + 0.25 * sqrt2 * x[t - 1, 4] + e[4]

        if system_id in {"system5", "system6"}:
            x[t, 1] += drift
            x[t, 3] += drift

    data = x[warmup + max_required_lag : warmup + max_required_lag + length]
    data = _zscore_channels(data)
    return data, paper_truth_adjacency()


def simulate_null_network(
    length: int,
    *,
    n_nodes: int = 5,
    warmup: int = 200,
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = length + warmup + 2
    data = np.zeros((total, n_nodes), dtype=float)

    for t in range(2, total):
        innovations = rng.normal(0.0, noise_scale, size=n_nodes)
        data[t] = 0.55 * data[t - 1] - 0.15 * data[t - 2] + innovations

    data = _zscore_channels(data[-length:])
    truth = np.zeros((n_nodes, n_nodes), dtype=bool)
    return data, truth


def simulate_sparse_var(config: SparseSimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    adjacency = rng.random((config.n_nodes, config.n_nodes)) < config.density
    np.fill_diagonal(adjacency, False)

    lag_weights = np.zeros((config.max_lag, config.n_nodes, config.n_nodes), dtype=float)

    for source in range(config.n_nodes):
        lag_weights[0, source, source] = rng.uniform(0.2, 0.45)
        if config.max_lag > 1:
            lag_weights[1, source, source] = rng.uniform(-0.15, 0.05)

    for source in range(config.n_nodes):
        for target in range(config.n_nodes):
            if not adjacency[source, target]:
                continue
            lag = int(rng.integers(1, config.max_lag + 1))
            weight = rng.uniform(0.15, 0.4) * rng.choice([-1.0, 1.0])
            lag_weights[lag - 1, target, source] = weight

    nonlinear_weights = np.zeros((config.n_nodes, config.n_nodes), dtype=float)
    if config.nonlinear:
        nonlinear_weights = adjacency.astype(float) * rng.uniform(0.05, 0.15, size=(config.n_nodes, config.n_nodes))

    drift_mask = np.zeros(config.n_nodes, dtype=float)
    if config.drift_strength > 0:
        drift_mask[rng.choice(config.n_nodes, size=max(1, config.n_nodes // 4), replace=False)] = 1.0

    driver_loadings = np.zeros(config.n_nodes, dtype=float)
    latent = np.zeros(config.length + config.warmup + config.max_lag + 1, dtype=float)
    if config.latent_common_driver:
        idx = rng.choice(config.n_nodes, size=max(2, config.n_nodes // 3), replace=False)
        driver_loadings[idx] = rng.uniform(0.1, 0.3, size=idx.size)

    total = config.length + config.warmup + config.max_lag + 1
    data = np.zeros((total, config.n_nodes), dtype=float)

    for t in range(config.max_lag, total):
        current = rng.normal(0.0, config.noise_scale, size=config.n_nodes)

        if config.latent_common_driver:
            latent[t] = 0.7 * latent[t - 1] - 0.1 * latent[t - 2] + rng.normal(0.0, config.noise_scale)
            current += driver_loadings * latent[t]

        for lag in range(1, config.max_lag + 1):
            coeff = lag_weights[lag - 1]
            if config.time_varying:
                coeff = coeff * (1.0 + 0.5 * np.sin(2 * np.pi * t / 200.0))
            current += coeff @ data[t - lag]

        if config.nonlinear:
            current += nonlinear_weights @ np.tanh(data[t - 1])

        if config.drift_strength > 0:
            current += drift_mask * config.drift_strength * (t - config.max_lag)

        data[t] = current

    data = _zscore_channels(data[config.warmup + config.max_lag : config.warmup + config.max_lag + config.length])
    return data, adjacency


def _zscore_channels(data: np.ndarray) -> np.ndarray:
    means = data.mean(axis=0, keepdims=True)
    stds = data.std(axis=0, keepdims=True)
    stds[stds == 0.0] = 1.0
    return (data - means) / stds
