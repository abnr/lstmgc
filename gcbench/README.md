# gcbench

`gcbench` is the starter codebase for the hybrid paper plan in [NEXT_STEPS.md](/mnt/hd/Artigos/bruno/NEXT_STEPS.md).

Its purpose is to support two parts of the paper:

- a synthetic benchmark for Granger-causality methods
- a real EEG application pipeline built around subject-level directed networks

Right now, the package is strongest on the benchmark side. It already supports:

- generation of the `system1` to `system8` synthetic scenarios from the manuscript draft
- a null simulator and a generic sparse-network simulator
- held-out Granger-style inference with unrestricted vs restricted regressors
- permutation-based edge testing and `BH-FDR`
- classical and neural model backends
- a CLI for listing experiments, scaffolding results folders, generating simulations, and running benchmarks
- basic EEG network-summary utilities for the application stage

The EEG preprocessing and full dataset ingestion pipeline are not implemented yet.

## Objectives

This package exists to answer questions like:

- When do neural Granger-causality models outperform classical VAR-style models?
- How do methods behave under nonlinearity, nonstationarity, lag mismatch, short recordings, or noise?
- Can the same inference framework be carried into a real EEG motor-task application?

The implementation is designed around the experimental plan documented in [NEXT_STEPS.md](/mnt/hd/Artigos/bruno/NEXT_STEPS.md).

## Project Layout

From the repository root:

```text
/mnt/hd/Artigos/bruno
├── gcbench/
│   ├── __init__.py
│   ├── __main__.py
│   ├── benchmark.py
│   ├── cli.py
│   ├── eeg.py
│   ├── metrics.py
│   ├── models.py
│   ├── registry.py
│   ├── simulations.py
│   ├── stats.py
│   └── README.md
├── tests/
│   └── test_smoke.py
├── NEXT_STEPS.md
└── venv/
```

## File Guide

### `gcbench/__main__.py`

Entry point for:

```bash
./venv/bin/python -m gcbench ...
```

### `gcbench/cli.py`

Command-line interface.

Current subcommands:

- `list-experiments`
- `scaffold-results`
- `simulate-paper-system`
- `run-benchmark`

### `gcbench/registry.py`

Registry of paper experiments from the benchmark and EEG application plan.

This is where the experiment IDs live:

- `B00` to `B14`
- `E00` to `E14`

### `gcbench/simulations.py`

Synthetic data generators.

Main functions:

- `generate_paper_system(...)`
- `simulate_null_network(...)`
- `simulate_sparse_var(...)`

The paper systems currently correspond to:

- `system1`: linear stationary fixed lags
- `system2`: linear with varying lag
- `system3`: nonlinear polynomial
- `system4`: nonlinear polynomial with varying lag
- `system5`: nonstationary drift
- `system6`: nonstationary drift with varying lag
- `system7`: time-varying sine coefficients
- `system8`: time-varying cosine coefficients

### `gcbench/models.py`

Model backends for unrestricted and restricted prediction.

Implemented classical models:

- `var`
- `ridge_var`

Implemented neural models:

- `simple_lstm`
- `conv_lstm`
- `rnn_gc`
- `neural_gc`

Notes:

- The neural models use PyTorch.
- The current `neural_gc` implementation is a feed-forward lag-window baseline, not a full reimplementation of all published Neural GC variants.
- The current `rnn_gc` implementation is a simple RNN regression baseline.

### `gcbench/benchmark.py`

Core benchmark runner.

Important functions:

- `split_time_series(...)`
- `run_granger_inference(...)`
- `run_paper_system_benchmark(...)`

This module is responsible for:

- train/validation/test splitting
- unrestricted vs restricted model fitting
- held-out MSE comparison
- permutation-based edge significance
- `BH-FDR`
- benchmark metric calculation

### `gcbench/stats.py`

Statistical helpers:

- block permutation
- permutation `p`-value computation
- Benjamini-Hochberg FDR control

### `gcbench/metrics.py`

Graph-recovery metrics:

- confusion counts
- precision
- sensitivity
- specificity
- `F1`
- false discovery rate
- `AUPRC`
- `AUROC`

### `gcbench/eeg.py`

Utilities for the application stage.

Current scope:

- validating epoch-array shapes
- concatenating preprocessed epochs
- averaging subject-level networks
- edge-frequency summaries
- graph-level summaries
- condition-difference summaries
- Jaccard-based reproducibility summaries

This file does not yet download, preprocess, or epoch EEG recordings by itself.

### `tests/test_smoke.py`

Smoke tests for:

- synthetic-system generation
- null-network generation
- classical end-to-end benchmark execution
- neural-model fit/predict path
- EEG summary utilities

## Environment Setup

The project now includes a local virtual environment in:

- [venv](/mnt/hd/Artigos/bruno/venv)

Use the interpreter from that environment for all `gcbench` commands:

```bash
./venv/bin/python --version
```

The current working environment includes:

- `numpy`
- `torch` (CPU-only)

If you ever need to recreate the environment:

```bash
python3 -m venv venv
./venv/bin/pip install numpy
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## How To Run

All commands below should be run from the repository root:

- `/mnt/hd/Artigos/bruno`

### 1. List the planned experiments

```bash
./venv/bin/python -m gcbench list-experiments
```

Only benchmark experiments:

```bash
./venv/bin/python -m gcbench list-experiments --domain benchmark
```

Only application experiments:

```bash
./venv/bin/python -m gcbench list-experiments --domain application
```

### 2. Scaffold a results directory

```bash
./venv/bin/python -m gcbench scaffold-results --base-dir results
```

This creates one folder per experiment ID, for example:

- `results/B00`
- `results/B01`
- `results/E00`
- `results/E01`

### 3. Generate one synthetic paper system

```bash
./venv/bin/python -m gcbench simulate-paper-system \
  --system system1 \
  --length 1200 \
  --seed 0 \
  --output results/B01/system1_seed0.npz
```

The `.npz` file contains:

- `data`: array shaped `(time, variables)`
- `truth`: adjacency matrix shaped `(variables, variables)`

### 4. Run a classical benchmark

Example with `VAR-GC`:

```bash
./venv/bin/python -m gcbench run-benchmark \
  --system system1 \
  --model var \
  --length 1200 \
  --max-lag 10 \
  --permutations 100 \
  --seed 0 \
  --output results/B01/var_system1_seed0.json
```

Example with regularized VAR:

```bash
./venv/bin/python -m gcbench run-benchmark \
  --system system3 \
  --model ridge_var \
  --length 1200 \
  --max-lag 10 \
  --permutations 100 \
  --seed 0 \
  --output results/B03/ridge_var_system3_seed0.json
```

### 5. Run a neural benchmark

Example with `Simple-LSTM`:

```bash
./venv/bin/python -m gcbench run-benchmark \
  --system system1 \
  --model simple_lstm \
  --length 180 \
  --max-lag 4 \
  --permutations 4 \
  --epochs 10 \
  --batch-size 16 \
  --hidden-size 8 \
  --patience 3 \
  --seed 2 \
  --output results/B01/simple_lstm_system1_seed2.json
```

Example with `Conv-LSTM`:

```bash
./venv/bin/python -m gcbench run-benchmark \
  --system system4 \
  --model conv_lstm \
  --length 300 \
  --max-lag 6 \
  --permutations 10 \
  --epochs 12 \
  --batch-size 16 \
  --hidden-size 12 \
  --patience 4 \
  --seed 1 \
  --output results/B04/conv_lstm_system4_seed1.json
```

Example with `RNN-GC`:

```bash
./venv/bin/python -m gcbench run-benchmark \
  --system system2 \
  --model rnn_gc \
  --length 300 \
  --max-lag 6 \
  --permutations 10 \
  --epochs 12 \
  --batch-size 16 \
  --hidden-size 12 \
  --patience 4 \
  --seed 1
```

Example with `Neural-GC`:

```bash
./venv/bin/python -m gcbench run-benchmark \
  --system system3 \
  --model neural_gc \
  --length 300 \
  --max-lag 6 \
  --permutations 10 \
  --epochs 12 \
  --batch-size 16 \
  --hidden-size 12 \
  --mlp-hidden 32 \
  --patience 4 \
  --seed 1
```

## CLI Reference

### `list-experiments`

Prints tab-separated experiment metadata:

- experiment ID
- domain
- priority
- owner
- dependencies
- title

### `scaffold-results`

Arguments:

- `--base-dir`: root folder for experiment directories, default `results`

### `simulate-paper-system`

Arguments:

- `--system`: one of `system1` to `system8`
- `--length`: number of returned time points
- `--seed`: random seed
- `--output`: output `.npz` path

### `run-benchmark`

Arguments:

- `--system`: one of `system1` to `system8`
- `--model`: one of `var`, `ridge_var`, `simple_lstm`, `conv_lstm`, `rnn_gc`, `neural_gc`
- `--length`: number of returned time points
- `--max-lag`: maximum lag window
- `--seed`: random seed
- `--train-fraction`: default `0.6`
- `--val-fraction`: default `0.2`
- `--permutations`: number of edgewise permutations
- `--block-size`: optional block size for time-preserving permutation
- `--q-value`: FDR threshold
- `--epochs`: neural-model epochs
- `--batch-size`: neural-model batch size
- `--hidden-size`: recurrent hidden size
- `--learning-rate`: optimizer learning rate
- `--patience`: early-stopping patience
- `--mlp-hidden`: hidden size for the feed-forward `neural_gc` model
- `--output`: optional output JSON path

## Output Format

`run-benchmark` prints a JSON object and optionally writes it to disk.

Important fields:

- `model_name`
- `max_lag`
- `runtime_seconds`
- `score_matrix`
- `pvalue_matrix`
- `significant_matrix`
- `unrestricted_mse`
- `metrics`

The `metrics` field currently includes:

- `tp`
- `tn`
- `fp`
- `fn`
- `precision`
- `sensitivity`
- `specificity`
- `f1`
- `false_discovery_rate`
- `auprc`
- `auroc`

## Programmatic Use

You can also import the package directly.

Example:

```python
from gcbench.benchmark import run_paper_system_benchmark

data, truth, result = run_paper_system_benchmark(
    "system1",
    model_name="simple_lstm",
    length=300,
    max_lag=6,
    n_permutations=10,
    random_state=0,
    model_kwargs={
        "epochs": 12,
        "batch_size": 16,
        "hidden_size": 12,
        "patience": 4,
    },
)

print(result.metrics)
```

## Testing

Run the smoke tests with system Python:

```bash
python3 -m unittest discover -s tests -v
```

Run them with the local virtual environment:

```bash
./venv/bin/python -m unittest discover -s tests -v
```

The `venv` version is the important one for neural models.

## EEG Utilities

The current EEG module assumes you already have preprocessed arrays.

Expected shape for epochs:

```text
(n_epochs, n_times, n_channels)
```

Example pattern:

```python
import numpy as np
from gcbench.eeg import concatenate_epochs, mean_network, condition_table

epochs = np.random.randn(20, 160, 12)
continuous = concatenate_epochs(epochs)
```

At this stage, `gcbench.eeg` is for post-preprocessing summaries, not for raw EEG ingestion.

## Current Limitations

- There is no batch runner for the full `B00-B14` and `E00-E14` matrix yet.
- There is no automatic downloader or preprocessor for PhysioNet EEG data yet.
- The neural models are functional research baselines, not final publication-grade tuned implementations.
- The current `neural_gc` and `rnn_gc` implementations are approximations for benchmarking, not strict replicas of every published method.
- Running unrestricted and restricted neural models for every edge is computationally expensive, especially on CPU.
- The benchmark currently works one synthetic system at a time from the CLI.

## Recommended Workflow

1. Read [NEXT_STEPS.md](/mnt/hd/Artigos/bruno/NEXT_STEPS.md) for the paper plan.
2. Scaffold the result directories.
3. Start with `B00-B04`.
4. Use small settings first to debug:
   - short `length`
   - few `permutations`
   - few `epochs`
5. Increase runtime settings only after outputs look sane.
6. Save every run to JSON so the paper tables can be built from disk.

## First Commands To Try

```bash
./venv/bin/python -m gcbench list-experiments --domain benchmark
./venv/bin/python -m gcbench scaffold-results --base-dir results
./venv/bin/python -m gcbench simulate-paper-system --system system1 --length 300 --seed 0 --output results/B01/system1_seed0.npz
./venv/bin/python -m gcbench run-benchmark --system system1 --model var --length 300 --max-lag 5 --permutations 8 --seed 0 --output results/B01/var_seed0.json
./venv/bin/python -m gcbench run-benchmark --system system1 --model simple_lstm --length 180 --max-lag 4 --permutations 4 --epochs 10 --batch-size 16 --hidden-size 8 --patience 3 --seed 0 --output results/B01/simple_lstm_seed0.json
```

## Next Development Targets

The most useful next steps for this package are:

- add batch orchestration for multiple seeds and multiple benchmark cells
- add result aggregation across seeds
- add plotting scripts for benchmark figures
- add EEG preprocessing and PhysioNet dataset loaders
- add experiment manifests so `B00-B14` can be launched directly by ID
