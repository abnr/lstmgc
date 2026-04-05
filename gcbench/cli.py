from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark import run_paper_system_benchmark
from .models import implemented_model_names, known_model_names
from .registry import ALL_EXPERIMENTS
from .simulations import PAPER_SYSTEMS, generate_paper_system


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and EEG utilities for the GC paper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cmd = subparsers.add_parser("list-experiments", help="Print the benchmark and application registry.")
    list_cmd.add_argument("--domain", choices=("all", "benchmark", "application"), default="all")

    scaffold_cmd = subparsers.add_parser("scaffold-results", help="Create experiment directories.")
    scaffold_cmd.add_argument("--base-dir", default="results")

    simulate_cmd = subparsers.add_parser("simulate-paper-system", help="Generate one of the paper's synthetic systems.")
    simulate_cmd.add_argument("--system", required=True, choices=sorted(PAPER_SYSTEMS))
    simulate_cmd.add_argument("--length", type=int, default=1200)
    simulate_cmd.add_argument("--seed", type=int, default=0)
    simulate_cmd.add_argument("--output", required=True)

    run_cmd = subparsers.add_parser("run-benchmark", help="Run a held-out GC benchmark on one synthetic paper system.")
    run_cmd.add_argument("--system", required=True, choices=sorted(PAPER_SYSTEMS))
    run_cmd.add_argument("--model", default="var", choices=known_model_names())
    run_cmd.add_argument("--length", type=int, default=1200)
    run_cmd.add_argument("--max-lag", type=int, default=10)
    run_cmd.add_argument("--seed", type=int, default=0)
    run_cmd.add_argument("--train-fraction", type=float, default=0.6)
    run_cmd.add_argument("--val-fraction", type=float, default=0.2)
    run_cmd.add_argument("--permutations", type=int, default=100)
    run_cmd.add_argument("--block-size", type=int, default=0)
    run_cmd.add_argument("--q-value", type=float, default=0.05)
    run_cmd.add_argument("--epochs", type=int, default=40)
    run_cmd.add_argument("--batch-size", type=int, default=64)
    run_cmd.add_argument("--hidden-size", type=int, default=15)
    run_cmd.add_argument("--learning-rate", type=float, default=1e-3)
    run_cmd.add_argument("--patience", type=int, default=8)
    run_cmd.add_argument("--mlp-hidden", type=int, default=64)
    run_cmd.add_argument("--output", help="Optional path for the JSON result.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-experiments":
        for spec in ALL_EXPERIMENTS:
            if args.domain != "all" and spec.domain != args.domain:
                continue
            depends_on = ",".join(spec.depends_on) if spec.depends_on else "-"
            print(f"{spec.id}\t{spec.domain}\t{spec.priority}\t{spec.owner}\t{depends_on}\t{spec.title}")
        return 0

    if args.command == "scaffold-results":
        base_dir = Path(args.base_dir)
        for spec in ALL_EXPERIMENTS:
            (base_dir / spec.id).mkdir(parents=True, exist_ok=True)
        print(f"Created result directories under {base_dir}")
        return 0

    if args.command == "simulate-paper-system":
        data, truth = generate_paper_system(args.system, args.length, seed=args.seed)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, data=data, truth=truth.astype(int))
        print(f"Saved {args.system} to {output}")
        return 0

    if args.command == "run-benchmark":
        if args.model not in implemented_model_names():
            parser.error(
                f"Model '{args.model}' is known but not implemented in this environment. "
                f"Implemented models: {', '.join(implemented_model_names())}."
            )
        _, _, result = run_paper_system_benchmark(
            args.system,
            model_name=args.model,
            length=args.length,
            max_lag=args.max_lag,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            n_permutations=args.permutations,
            block_size=args.block_size or None,
            q_value=args.q_value,
            random_state=args.seed,
            model_kwargs={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "hidden_size": args.hidden_size,
                "learning_rate": args.learning_rate,
                "patience": args.patience,
                "mlp_hidden": args.mlp_hidden,
            },
        )
        payload = result.to_dict()
        print(json.dumps(payload, indent=2))
        if args.output:
            result.save_json(args.output)
        return 0

    parser.error("Unknown command")
    return 2
