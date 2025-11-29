"""
pi_monte_carlo.py

Monte Carlo estimators for pi. This module supports the project tasks that
require a statistical estimate of pi and an investigation of the resulting
distribution and errors.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_pi_unit_circle(
    number_of_samples: int,
    random_number_generator: np.random.Generator | None = None,
) -> float:
    """Estimate pi using the unit-circle-in-a-square method.

    Points are drawn uniformly in the square [-1, 1] x [-1, 1]. The fraction
    of points inside the unit circle is used to estimate pi.

    Args:
        number_of_samples (int): Number of random points to generate.
        random_number_generator (np.random.Generator | None): Optional NumPy
            random number generator instance. If None, a default generator is
            created.

    Returns:
        float: Monte Carlo estimate of pi.
    """
    if number_of_samples <= 0:
        raise ValueError("number_of_samples must be positive.")

    if random_number_generator is None:
        random_number_generator = np.random.default_rng()

    points = random_number_generator.uniform(low=-1.0, high=1.0, size=(number_of_samples, 2))
    squared_radii = np.sum(points ** 2, axis=1)
    number_inside_circle = np.count_nonzero(squared_radii <= 1.0)

    fraction_inside = number_inside_circle / float(number_of_samples)
    pi_estimate = 4.0 * fraction_inside
    return pi_estimate


def run_pi_experiment(
    number_of_samples: int,
    number_of_repeats: int,
    seed: int | None = None,
) -> Tuple[np.ndarray, float, float]:
    """Run repeated Monte Carlo estimates of pi for statistical analysis.

    Args:
        number_of_samples (int): Number of random points per estimate.
        number_of_repeats (int): Number of independent estimates to generate.
        seed (int | None): Optional seed for reproducibility.

    Returns:
        Tuple[np.ndarray, float, float]:
            - Array of individual pi estimates of shape (number_of_repeats,).
            - Mean of the estimates.
            - Standard deviation of the estimates.
    """
    if number_of_repeats <= 0:
        raise ValueError("number_of_repeats must be positive.")

    random_number_generator = np.random.default_rng(seed)

    pi_estimates = np.empty(number_of_repeats, dtype=float)
    for repeat_index in range(number_of_repeats):
        pi_estimates[repeat_index] = estimate_pi_unit_circle(
            number_of_samples=number_of_samples,
            random_number_generator=random_number_generator,
        )

    mean_estimate = float(np.mean(pi_estimates))
    standard_deviation = float(np.std(pi_estimates, ddof=1))
    return pi_estimates, mean_estimate, standard_deviation
