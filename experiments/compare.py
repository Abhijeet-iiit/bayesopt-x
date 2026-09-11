"""Compare Bayesian Optimization against Random Search on 1-D Sphere."""

from __future__ import annotations

import argparse

import numpy as np

from experiments.cli import add_optimization_arguments, configure_stdio
from experiments.config import PLOTS_DIR, total_evaluations
from experiments.plotting import plt, save_current_figure
from src.black_box import sphere
from src.optimizer import BayesianOptimizer
from src.random_search import RandomSearch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Bayesian Optimization with Random Search.",
    )
    add_optimization_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Run the optimization comparison and save a convergence plot."""
    configure_stdio()
    args = parse_args()
    bounds = np.array([[-5.0, 5.0]])
    n_evaluations = total_evaluations(args.n_initial, args.n_iterations)

    bayesian_optimizer = BayesianOptimizer(
        objective=sphere,
        bounds=bounds,
        acquisition=args.acquisition,
        minimize=True,
        random_state=args.seed,
    )
    bayesian_x, bayesian_y = bayesian_optimizer.optimize(
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
    )

    random_optimizer = RandomSearch(
        objective=sphere,
        bounds=bounds,
        minimize=True,
        random_state=args.seed,
    )
    random_x, random_y = random_optimizer.optimize(n_iterations=n_evaluations)

    print("=" * 60)
    print("Bayesian Optimization vs Random Search")
    print("=" * 60)
    print("\nBayesian Optimization")
    print(f"Best x: {bayesian_x}")
    print(f"Best y: {bayesian_y:.10f}")
    print("\nRandom Search")
    print(f"Best x: {random_x}")
    print(f"Best y: {random_y:.10f}")
    print("\nTrue optimum")
    print("x = [0.0]")
    print("y = 0.0")

    plt.figure(figsize=(9, 5))
    plt.plot(
        np.arange(1, len(bayesian_optimizer.history) + 1),
        bayesian_optimizer.history,
        marker="o",
        label="Bayesian Optimization",
    )
    plt.plot(
        np.arange(1, len(random_optimizer.history) + 1),
        random_optimizer.history,
        marker="s",
        label="Random Search",
    )
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Objective Value")
    plt.title("Bayesian Optimization vs Random Search (Sphere)")
    plt.legend()
    plt.grid(True)

    output_path = PLOTS_DIR / "optimization_comparison.png"
    save_current_figure(output_path)
    print(f"\nComparison plot saved to: {output_path}")


if __name__ == "__main__":
    main()
