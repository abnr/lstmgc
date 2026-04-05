"""Utilities for the hybrid benchmark + EEG application paper."""

from .benchmark import BenchmarkResult, run_granger_inference, run_paper_system_benchmark
from .registry import APPLICATION_EXPERIMENTS, BENCHMARK_EXPERIMENTS, get_experiment
from .simulations import generate_paper_system, simulate_null_network, simulate_sparse_var

__all__ = [
    "APPLICATION_EXPERIMENTS",
    "BENCHMARK_EXPERIMENTS",
    "BenchmarkResult",
    "generate_paper_system",
    "get_experiment",
    "run_granger_inference",
    "run_paper_system_benchmark",
    "simulate_null_network",
    "simulate_sparse_var",
]
