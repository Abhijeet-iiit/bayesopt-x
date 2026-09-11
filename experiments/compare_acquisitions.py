"""Compare Random Search, Expected Improvement, and UCB."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.cli import add_optimization_arguments, configure_stdio
from experiments.config import resolve_benchmarks, total_evaluations
from experiments.runners import run_bayesian_optimization, run_random_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Expected Improvement, UCB, and Random Search.",
    )
    add_optimization_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Run the complete acquisition-function comparison."""
    configure_stdio()
    args = parse_args()
    n_evaluations = total_evaluations(args.n_initial, args.n_iterations)
    benchmarks = resolve_benchmarks(args.benchmark)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    print("=" * 80)
    print("BAYESOPT-X: ACQUISITION FUNCTION COMPARISON")
    print("=" * 80)
    print(f"Seeds: {args.seeds}")
    print(f"Evaluations per run: {n_evaluations}")

    for spec in benchmarks:
        print(f"\n{'-' * 80}")
        print(f"Benchmark: {spec.name}")
        print(f"{'-' * 80}")

        for seed in args.seeds:
            _, ei_result = run_bayesian_optimization(
                spec.function,
                spec.bounds,
                seed=seed,
                acquisition="ei",
                n_initial=args.n_initial,
                n_iterations=args.n_iterations,
            )
            _, ucb_result = run_bayesian_optimization(
                spec.function,
                spec.bounds,
                seed=seed,
                acquisition="ucb",
                n_initial=args.n_initial,
                n_iterations=args.n_iterations,
            )
            _, random_result = run_random_search(
                spec.function,
                spec.bounds,
                seed=seed,
                n_iterations=n_evaluations,
            )

            results.extend(
                [
                    {
                        "benchmark": spec.name,
                        "seed": seed,
                        "algorithm": "Expected Improvement",
                        "best_objective": ei_result,
                    },
                    {
                        "benchmark": spec.name,
                        "seed": seed,
                        "algorithm": "UCB",
                        "best_objective": ucb_result,
                    },
                    {
                        "benchmark": spec.name,
                        "seed": seed,
                        "algorithm": "Random Search",
                        "best_objective": random_result,
                    },
                ]
            )
            print(
                f"Seed {seed:4d} | "
                f"EI: {ei_result:10.5f} | "
                f"UCB: {ucb_result:10.5f} | "
                f"Random: {random_result:10.5f}"
            )

    results_df = pd.DataFrame(results)
    raw_path = output_dir / "acquisition_results.csv"
    results_df.to_csv(raw_path, index=False)

    summary = (
        results_df.groupby(["benchmark", "algorithm"], sort=True)["best_objective"]
        .agg(["mean", "std", "min"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_best_objective",
                "std": "std_best_objective",
                "min": "best_observed",
            }
        )
    )
    summary_path = output_dir / "acquisition_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for spec in benchmarks:
        print(f"\n{spec.name}")
        print("-" * 80)
        benchmark_summary = summary[summary["benchmark"] == spec.name]
        for _, row in benchmark_summary.iterrows():
            print(
                f"{row['algorithm']:22s} | "
                f"Mean: {row['mean_best_objective']:10.5f} | "
                f"Std: {row['std_best_objective']:10.5f} | "
                f"Best: {row['best_observed']:10.5f}"
            )

    print("\n" + "=" * 80)
    print(f"Raw results saved to: {raw_path}")
    print(f"Summary saved to:     {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
