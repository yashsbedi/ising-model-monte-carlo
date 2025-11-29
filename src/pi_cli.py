"""
pi_cli.py

Command-line interface for a basic Monte Carlo estimation of pi using
uniform random points in the unit square.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from .pi_monte_carlo import estimate_pi_unit_circle


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the pi estimation script.

    Returns:
        argparse.Namespace: Parsed arguments containing number_of_samples
        and optional random_seed.
    """
    parser = argparse.ArgumentParser(
        description="Monte Carlo estimator for pi using random points in a square."
    )
    parser.add_argument(
        "--number_of_samples",
        type=int,
        default=100_000,
        help="Number of random samples to use (default: 100000).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional seed for the random number generator.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a single Monte Carlo estimate of pi and print basic diagnostics."""
    arguments = parse_arguments()

    if arguments.random_seed is not None:
        np.random.seed(arguments.random_seed)

    pi_estimate = estimate_pi_unit_circle(
        number_of_samples=arguments.number_of_samples,
    )
    absolute_error = abs(pi_estimate - math.pi)

    print(f"Number of samples : {arguments.number_of_samples}")
    print(f"Estimated pi      : {pi_estimate:.6f}")
    print(f"Absolute error    : {absolute_error:.6e}")


if __name__ == "__main__":
    main()
