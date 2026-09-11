"""Run a basic Bayesian Optimization experiment on the Sphere function."""

from __future__ import annotations

import argparse

import numpy as np

from experiments.cli import add_optimization_arguments, configure_stdio
from experiments.config import PLOTS_DIR
from experiments.plotting import plt, save_current_figure
from src.black_box import sphere
from src.optimizer import BayesianOptimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization on the 1-D Sphere function.",
    )
    add_optimization_arguments(parser)
    return parser.parse_args()


def main() -> None:
    """Run the Sphere benchmark and save a convergence plot."""
    configure_stdio()
    args = parse_args()
    bounds = np.array([[-5.0, 5.0]])

    optimizer = BayesianOptimizer(
        objective=sphere,
        bounds=bounds,
        acquisition=args.acquisition,
        minimize=True,
        random_state=args.seed,
    )
    best_x, best_y = optimizer.optimize(
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
    )

    print("=" * 50)
    print("Bayesian Optimization - Sphere Function")
    print("=" * 50)
    print(f"Best x: {best_x}")
    print(f"Best y: {best_y:.10f}")
    print(f"Total evaluations: {len(optimizer.y)}")
    print(f"Seed: {args.seed}")
    print(f"Acquisition: {args.acquisition}")
    print("=" * 50)

    evaluations = np.arange(1, len(optimizer.history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(evaluations, optimizer.history, marker="o")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Best Objective Value")
    plt.title("Bayesian Optimization Convergence (Sphere)")
    plt.grid(True)

    output_path = PLOTS_DIR / "sphere_convergence.png"
    save_current_figure(output_path)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
