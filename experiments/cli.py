"""Command-line argument helpers for experiment scripts."""

from __future__ import annotations

import argparse
import sys

from .config import (
    ACQUISITION_CHOICES,
    BENCHMARK_CHOICES,
    DEFAULT_N_INITIAL,
    DEFAULT_N_ITERATIONS,
    DEFAULT_SEEDS,
    RESULTS_DIR,
)


def configure_stdio() -> None:
    """Keep CLI output from buffering until the script finishes."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


def add_optimization_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared budget, seed, benchmark, and acquisition flags."""
    parser.add_argument("--n-initial", type=int, default=DEFAULT_N_INITIAL)
    parser.add_argument("--n-iterations", type=int, default=DEFAULT_N_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--benchmark",
        choices=[*BENCHMARK_CHOICES, "all"],
        default="all",
    )
    parser.add_argument(
        "--acquisition",
        choices=ACQUISITION_CHOICES,
        default="ei",
    )
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
