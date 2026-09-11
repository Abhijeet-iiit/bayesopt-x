"""Run Bayesian Optimization and Random Search on multiple benchmarks."""

from __future__ import annotations

import argparse

from experiments.cli import add_optimization_arguments, configure_stdio
from experiments.config import resolve_benchmarks, total_evaluations
from experiments.runners import run_bayesian_optimization, run_random_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Bayesian Optimization and Random Search "
        "on one or more benchmark functions (single seed).",
    )
    add_optimization_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Run all requested benchmark experiments."""
    configure_stdio()
    args = parse_args()
    n_evaluations = total_evaluations(args.n_initial, args.n_iterations)
    benchmarks = resolve_benchmarks(args.benchmark)

    print("\n" + "=" * 70)
    print("BAYESIAN OPTIMIZATION vs RANDOM SEARCH")
    print("=" * 70)
    print(f"Seed: {args.seed}")
    print(f"Acquisition: {args.acquisition}")
    print(f"Evaluations: {n_evaluations}")

    for spec in benchmarks:
        bayesian_x, bayesian_y = run_bayesian_optimization(
            spec.function,
            spec.bounds,
            seed=args.seed,
            acquisition=args.acquisition,
            n_initial=args.n_initial,
            n_iterations=args.n_iterations,
        )
        random_x, random_y = run_random_search(
            spec.function,
            spec.bounds,
            seed=args.seed,
            n_iterations=n_evaluations,
        )

        print(f"\n{spec.name}")
        print("-" * 70)
        print("Bayesian Optimization:")
        print(f"  Best x: {bayesian_x}")
        print(f"  Best y: {bayesian_y:.8f}")
        print("Random Search:")
        print(f"  Best x: {random_x}")
        print(f"  Best y: {random_y:.8f}")

        if bayesian_y < random_y:
            print("  Winner: Bayesian Optimization")
        elif random_y < bayesian_y:
            print("  Winner: Random Search")
        else:
            print("  Result: Tie")


if __name__ == "__main__":
    main()
