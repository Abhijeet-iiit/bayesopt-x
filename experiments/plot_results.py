"""Generate plots from Bayesian Optimization benchmark CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.cli import configure_stdio
from experiments.config import PLOTS_DIR, RESULTS_DIR
from experiments.plotting import plt, save_current_figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from experiment CSV files.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory containing experiment CSV files.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(PLOTS_DIR),
        help="Directory where plots will be written.",
    )
    return parser.parse_args()


def _plot_algorithm_bars(
    summary: pd.DataFrame,
    plots_dir: Path,
    filename_suffix: str,
    overall_title: str,
    overall_filename: str,
) -> None:
    for benchmark in summary["benchmark"].unique():
        benchmark_data = summary[summary["benchmark"] == benchmark]
        plt.figure(figsize=(8, 5))
        plt.bar(benchmark_data["algorithm"], benchmark_data["best_objective"])
        plt.ylabel("Mean Best Objective (lower is better)")
        plt.xlabel("Algorithm")
        plt.title(f"{benchmark}: Mean Best Objective")
        plt.xticks(rotation=15)
        output_path = plots_dir / f"{str(benchmark).lower()}_{filename_suffix}.png"
        save_current_figure(output_path)
        print(f"Saved: {output_path}")

    ax = summary.pivot(
        index="benchmark",
        columns="algorithm",
        values="best_objective",
    ).plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Mean Best Objective (lower is better)")
    ax.set_xlabel("Benchmark Function")
    ax.set_title(overall_title)
    plt.xticks(rotation=0)
    overall_path = plots_dir / overall_filename
    save_current_figure(overall_path)
    print(f"Saved: {overall_path}")


def main() -> None:
    """Generate benchmark comparison plots from saved CSV files."""
    configure_stdio()
    args = parse_args()
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    acquisition_path = results_dir / "acquisition_results.csv"
    if acquisition_path.exists():
        df = pd.read_csv(acquisition_path)
        summary = (
            df.groupby(["benchmark", "algorithm"], sort=True)["best_objective"]
            .mean()
            .reset_index()
        )
        _plot_algorithm_bars(
            summary,
            plots_dir,
            filename_suffix="comparison",
            overall_title="Acquisition Functions vs Random Search",
            overall_filename="overall_comparison.png",
        )
    else:
        print(f"Skipping acquisition plots; missing {acquisition_path}")

    benchmark_path = results_dir / "benchmark_results.csv"
    if benchmark_path.exists():
        df = pd.read_csv(benchmark_path)
        summary = (
            df.groupby(["benchmark", "algorithm"], sort=True)["best_objective"]
            .mean()
            .reset_index()
        )
        _plot_algorithm_bars(
            summary,
            plots_dir,
            filename_suffix="benchmark_comparison",
            overall_title="Bayesian Optimization vs Random Search",
            overall_filename="benchmark_overall_comparison.png",
        )
    else:
        print(f"Skipping benchmark plots; missing {benchmark_path}")


if __name__ == "__main__":
    main()
