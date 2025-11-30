"""
ising2d_correlation_length.py

Estimate the correlation length of the 2D ferromagnetic Ising model at a
single temperature using the spin-spin correlation function along one
lattice direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .ising2d import Ising2DModel
from .metropolis import metropolis_single_flip_2d


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the correlation length script.

    Returns:
        argparse.Namespace: Parsed arguments, including lattice size,
        temperature, and simulation parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the correlation length of the 2D Ising model at a "
            "single temperature using spin-spin correlations."
        )
    )
    parser.add_argument(
        "--lattice_size",
        type=int,
        default=32,
        help="Linear lattice size L for an LxL system (default: 32).",
    )
    parser.add_argument(
        "--coupling_constant",
        type=float,
        default=1.0,
        help="Nearest neighbour coupling constant J (default: 1.0).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.3,
        help="Temperature at which to estimate the correlation length "
        "(default: 2.3).",
    )
    parser.add_argument(
        "--number_of_sweeps",
        type=int,
        default=50_000,
        help="Total number of Monte Carlo sweeps (default: 50000).",
    )
    parser.add_argument(
        "--burn_in_sweeps",
        type=int,
        default=10_000,
        help="Number of sweeps discarded as burn-in (default: 10000).",
    )
    parser.add_argument(
        "--sampling_interval",
        type=int,
        default=50,
        help="Interval in sweeps between correlation measurements "
        "(default: 50).",
    )
    parser.add_argument(
        "--maximum_distance",
        type=int,
        default=10,
        help="Maximum separation r for the correlation function C(r) "
        "(default: 10).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="results/figures",
        help=(
            "Directory where correlation plots will be saved as PNG files "
            "(default: results/figures)."
        ),
    )
    return parser.parse_args()


def run_ising2d_equilibrated_samples(
    lattice_size: int,
    temperature: float,
    coupling_constant: float,
    number_of_sweeps: int,
    burn_in_sweeps: int,
    sampling_interval: int,
    random_seed: int | None,
) -> np.ndarray:
    """Run a 2D Ising simulation and return sampled configurations.

    Args:
        lattice_size (int): Linear lattice size L.
        temperature (float): Simulation temperature T.
        coupling_constant (float): Coupling constant J.
        number_of_sweeps (int): Total number of sweeps to perform.
        burn_in_sweeps (int): Number of sweeps discarded as burn-in.
        sampling_interval (int): Interval in sweeps between stored samples.
        random_seed (int | None): Optional random seed.

    Returns:
        np.ndarray: Array of shape (number_of_samples, L, L) containing
        sampled spin configurations.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    model = Ising2DModel(
        lattice_size=lattice_size,
        coupling_constant=coupling_constant,
    )

    number_of_samples = max(
        0, (number_of_sweeps - burn_in_sweeps) // sampling_interval
    )
    samples = np.empty(
        (number_of_samples, lattice_size, lattice_size), dtype=int
    )

    sample_index = 0
    for sweep_index in range(number_of_sweeps):
        metropolis_single_flip_2d(model, inverse_temperature=1.0 / temperature)

        if (
            sweep_index >= burn_in_sweeps
            and (sweep_index - burn_in_sweeps) % sampling_interval == 0
        ):
            samples[sample_index] = model.spins.copy()
            sample_index += 1

    return samples


def compute_correlation_function(
    spin_samples: np.ndarray,
    maximum_distance: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the spin-spin correlation function C(r) along one direction.

    Args:
        spin_samples (np.ndarray): Array of shape (S, L, L) containing S
            sampled spin configurations.
        maximum_distance (int): Maximum distance r for C(r).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Distances r as an array of integers.
            - Mean correlation values C(r).
    """
    number_of_samples, lattice_size, _ = spin_samples.shape
    maximum_distance = min(maximum_distance, lattice_size // 2)

    distances = np.arange(0, maximum_distance + 1, dtype=int)
    correlations = np.zeros_like(distances, dtype=float)

    for configuration in spin_samples:
        # Choose all starting sites along one row for better averaging
        row_index = 0
        for column_index in range(lattice_size):
            s0 = configuration[row_index, column_index]
            for r_index, distance in enumerate(distances):
                neighbour_column = (column_index + distance) % lattice_size
                s_r = configuration[row_index, neighbour_column]
                correlations[r_index] += s0 * s_r

    number_of_start_sites = lattice_size
    normalisation = float(number_of_samples * number_of_start_sites)
    correlations /= normalisation

    return distances.astype(float), correlations


def estimate_correlation_length(
    distances: np.ndarray,
    correlations: np.ndarray,
    minimum_distance: int = 1,
    maximum_distance_fit: int | None = None,
) -> float:
    """Estimate the correlation length from the decay of C(r).

    Performs a linear least-squares fit to

        ln C(r) = a - r / xi

    over a chosen range of distances.

    Args:
        distances (np.ndarray): Distances r.
        correlations (np.ndarray): Correlation values C(r).
        minimum_distance (int): Smallest r to include in the fit
            (default: 1).
        maximum_distance_fit (int | None): Largest r to include in the fit.
            If None, all available distances beyond minimum_distance are used.

    Returns:
        float: Estimated correlation length xi. Returns np.nan if the fit
        cannot be performed.
    """
    if maximum_distance_fit is None:
        maximum_distance_fit = int(distances[-1])

    mask = (distances >= minimum_distance) & (
        distances <= float(maximum_distance_fit)
    )
    fit_distances = distances[mask]
    fit_correlations = correlations[mask]

    positive_mask = fit_correlations > 0.0
    fit_distances = fit_distances[positive_mask]
    fit_correlations = fit_correlations[positive_mask]

    if len(fit_distances) < 2:
        return float("nan")

    log_correlations = np.log(fit_correlations)
    coefficients = np.polyfit(fit_distances, log_correlations, deg=1)
    slope = coefficients[0]
    if slope >= 0.0:
        return float("nan")

    xi = -1.0 / slope
    return float(xi)


def make_correlation_plots(
    distances: np.ndarray,
    correlations: np.ndarray,
    xi_estimate: float,
    temperature: float,
    output_directory: str,
) -> None:
    """Create and save plots of C(r) and the fitted exponential.

    Args:
        distances (np.ndarray): Distances r.
        correlations (np.ndarray): Correlation values C(r).
        xi_estimate (float): Estimated correlation length.
        temperature (float): Simulation temperature.
        output_directory (str): Directory where plots will be saved.
    """
    output_dir_path = Path(output_directory)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(distances, correlations, marker="o", linestyle="none", label="C(r) data")

    if np.isfinite(xi_estimate):
        fitted = np.exp(-distances / xi_estimate)
        plt.plot(
            distances,
            fitted,
            label=f"exp(-r / xi), xi = {xi_estimate:.2f}",
        )

    plt.xlabel("Distance r (lattice units)")
    plt.ylabel("Correlation C(r)")
    plt.title(f"2D Ising model: spin-spin correlation at T = {temperature:.3f}")
    plt.legend()

    output_path = (
        output_dir_path
        / f"ising2d_correlation_function_T_{temperature:.3f}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved correlation plot to {output_path}")


def main() -> None:
    """Entry point for the 2D Ising correlation length script."""
    arguments = parse_arguments()

    spin_samples = run_ising2d_equilibrated_samples(
        lattice_size=arguments.lattice_size,
        temperature=arguments.temperature,
        coupling_constant=arguments.coupling_constant,
        number_of_sweeps=arguments.number_of_sweeps,
        burn_in_sweeps=arguments.burn_in_sweeps,
        sampling_interval=arguments.sampling_interval,
        random_seed=arguments.random_seed,
    )

    distances, correlations = compute_correlation_function(
        spin_samples=spin_samples,
        maximum_distance=arguments.maximum_distance,
    )

    xi_estimate = estimate_correlation_length(
        distances=distances,
        correlations=correlations,
        minimum_distance=1,
        maximum_distance_fit=arguments.maximum_distance,
    )

    print(
        f"Estimated correlation length at T = {arguments.temperature:.3f}: "
        f"xi = {xi_estimate:.3f}"
    )

    make_correlation_plots(
        distances=distances,
        correlations=correlations,
        xi_estimate=xi_estimate,
        temperature=arguments.temperature,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
