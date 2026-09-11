"""BayesOpt-X: Bayesian Optimization for expensive black-box functions."""

from .acquisition import expected_improvement, upper_confidence_bound
from .black_box import BENCHMARK_FUNCTIONS, rastrigin, rosenbrock, sphere
from .gp_model import GaussianProcessSurrogate
from .optimizer import BayesianOptimizer
from .random_search import RandomSearch

__all__ = [
    "BENCHMARK_FUNCTIONS",
    "BayesianOptimizer",
    "GaussianProcessSurrogate",
    "RandomSearch",
    "expected_improvement",
    "rastrigin",
    "rosenbrock",
    "sphere",
    "upper_confidence_bound",
]

__version__ = "0.1.0"
