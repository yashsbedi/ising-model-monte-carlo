"""
pi_statistics.py

Command-line interface for running repeated Monte Carlo estimates of pi and
saving the results for later analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .pi_monte_carlo import run_pi_experiment


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the pi statistics script.

    Returns:
        argparse.Namespace: Parsed arguments including number_of_samples,
        number_of_repeats, random_seed, and output_path.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated Monte Carlo estimates of pi and store the results "
            "for statistical analysis."
        )
    )
    parser.add_argument(
        "--number_of_samples",
        type=int,
        default=10_000,
        help="Number of random points per estimate (default: 10000).",
    )
    parser.add_argument(
        "--number_of_repeats",
        type=int,
        default=100,
        help="Number of independent estimates to generate (default: 100).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional seed for the random number generator.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/raw/pi_estimates.npy",
        help=(
            "Path to a NumPy .npy file where the estimates will be stored "
            "(default: results/raw/pi_estimates.npy)."
        ),
    )
    return parser.parse_args()


def run_and_save_pi_statistics(
    number_of_samples: int,
    number_of_repeats: int,
    random_seed: int | None,
    output_path: str,
) -> Tuple[np.ndarray, float, float]:
    """Run repeated pi estimates and save them to disk.

    Args:
        number_of_samples (int): Number of random points per estimate.
        number_of_repeats (int): Number of independent estimates.
        random_seed (int | None): Optional random seed.
        output_path (str): File path for saving the array of estimates.

    Returns:
        Tuple[np.ndarray, float, float]:
            - Array of pi estimates.
            - Sample mean of the estimates.
            - Sample standard deviation of the estimates.
    """
    estimates, mean_estimate, standard_deviation = run_pi_experiment(
        number_of_samples=number_of_samples,
        number_of_repeats=number_of_repeats,
        seed=random_seed,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, estimates)

    return estimates, mean_estimate, standard_deviation


def main() -> None:
    """Entry point for generating and saving pi estimator statistics."""
    arguments = parse_arguments()

    estimates, mean_estimate, standard_deviation = run_and_save_pi_statistics(
        number_of_samples=arguments.number_of_samples,
        number_of_repeats=arguments.number_of_repeats,
        random_seed=arguments.random_seed,
        output_path=arguments.output_path,
    )

    print(f"Saved {len(estimates)} estimates to {arguments.output_path}")
    print(f"Sample mean of estimates : {mean_estimate:.6f}")
    print(f"Sample standard deviation: {standard_deviation:.6e}")


if __name__ == "__main__":
    main()
