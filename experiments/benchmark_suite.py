"""Statistical benchmark suite for BayesOpt-X.

Compares Bayesian Optimization against Random Search across multiple
benchmark functions and random seeds, then writes CSV summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.cli import add_optimization_arguments, configure_stdio
from experiments.config import resolve_benchmarks, total_evaluations
from experiments.runners import run_bayesian_optimization, run_random_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed Bayesian Optimization vs Random Search suite.",
    )
    add_optimization_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Run the complete benchmark suite and save CSV results."""
    configure_stdio()
    args = parse_args()
    n_evaluations = total_evaluations(args.n_initial, args.n_iterations)
    benchmarks = resolve_benchmarks(args.benchmark)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    print("=" * 75)
    print("BAYESOPT-X STATISTICAL BENCHMARK")
    print("=" * 75)
    print(f"Seeds: {args.seeds}")
    print(f"Evaluations per run: {n_evaluations}")
    print(f"Acquisition: {args.acquisition}")

    for spec in benchmarks:
        print(f"\n{'-' * 75}")
        print(f"Benchmark: {spec.name}")
        print(f"{'-' * 75}")

        for seed in args.seeds:
            _, bayesian_y = run_bayesian_optimization(
                spec.function,
                spec.bounds,
                seed=seed,
                acquisition=args.acquisition,
                n_initial=args.n_initial,
                n_iterations=args.n_iterations,
            )
            _, random_y = run_random_search(
                spec.function,
                spec.bounds,
                seed=seed,
                n_iterations=n_evaluations,
            )

            results.append(
                {
                    "benchmark": spec.name,
                    "seed": seed,
                    "algorithm": "Bayesian Optimization",
                    "acquisition": args.acquisition,
                    "best_objective": bayesian_y,
                }
            )
            results.append(
                {
                    "benchmark": spec.name,
                    "seed": seed,
                    "algorithm": "Random Search",
                    "acquisition": "n/a",
                    "best_objective": random_y,
                }
            )

            if bayesian_y < random_y:
                winner = "Bayesian Optimization"
            elif random_y < bayesian_y:
                winner = "Random Search"
            else:
                winner = "Tie"
            print(
                f"Seed {seed:4d} | "
                f"Bayesian: {bayesian_y:12.6f} | "
                f"Random: {random_y:12.6f} | "
                f"Winner: {winner}"
            )

    results_df = pd.DataFrame(results)
    raw_path = output_dir / "benchmark_results.csv"
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
    summary_path = output_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    for spec in benchmarks:
        benchmark_results = results_df[results_df["benchmark"] == spec.name]
        bayesian_values = benchmark_results.loc[
            benchmark_results["algorithm"] == "Bayesian Optimization",
            "best_objective",
        ].to_numpy()
        random_values = benchmark_results.loc[
            benchmark_results["algorithm"] == "Random Search",
            "best_objective",
        ].to_numpy()
        bayesian_wins = int(np.sum(bayesian_values < random_values))
        random_wins = int(np.sum(random_values < bayesian_values))
        ties = int(np.sum(bayesian_values == random_values))

        print(f"\n{spec.name}")
        print(f"  Bayesian mean: {np.mean(bayesian_values):.6f}")
        print(f"  Bayesian std:  {np.std(bayesian_values, ddof=1):.6f}")
        print(f"  Random mean:   {np.mean(random_values):.6f}")
        print(f"  Random std:    {np.std(random_values, ddof=1):.6f}")
        print(
            f"  Record: Bayesian {bayesian_wins} / "
            f"Random {random_wins} / Ties {ties} "
            f"(out of {len(args.seeds)})"
        )

    print("\n" + "=" * 75)
    print(f"Raw results saved to: {raw_path}")
    print(f"Summary saved to:     {summary_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
