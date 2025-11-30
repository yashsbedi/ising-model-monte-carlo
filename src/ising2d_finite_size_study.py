"""
ising2d_finite_size_study.py

Finite-size study of the 2D ferromagnetic Ising model. Runs the
temperature sweep for several lattice sizes and plots the specific heat
and susceptibility as functions of temperature for comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .ising2d_temperature_sweep import run_temperature_sweep


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the finite-size study script.

    Returns:
        argparse.Namespace: Parsed arguments including the list of lattice
        sizes and simulation parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run 2D ferromagnetic Ising temperature sweeps for several "
            "lattice sizes and compare specific heat and susceptibility."
        )
    )
    parser.add_argument(
        "--lattice_sizes",
        type=int,
        nargs="+",
        default=[8, 16, 24],
        help="List of lattice sizes L to study (default: 8 16 24).",
    )
    parser.add_argument(
        "--coupling_constant",
        type=float,
        default=1.0,
        help="Nearest neighbour coupling constant J (default: 1.0).",
    )
    parser.add_argument(
        "--temperature_min",
        type=float,
        default=0.01,
        help="Minimum temperature in the sweep (default: 0.01).",
    )
    parser.add_argument(
        "--temperature_max",
        type=float,
        default=4.0,
        help="Maximum temperature in the sweep (default: 4.0).",
    )
    parser.add_argument(
        "--number_of_temperatures",
        type=int,
        default=20,
        help="Number of temperature points in each sweep (default: 20).",
    )
    parser.add_argument(
        "--number_of_sweeps",
        type=int,
        default=20_000,
        help="Total number of Monte Carlo sweeps per temperature (default: 20000).",
    )
    parser.add_argument(
        "--burn_in_sweeps",
        type=int,
        default=5_000,
        help="Number of initial sweeps discarded as burn-in (default: 5000).",
    )
    parser.add_argument(
        "--sampling_interval",
        type=int,
        default=10,
        help="Record observables every this many sweeps (default: 10).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Optional base seed. Different sizes and temperatures use offsets.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="results/figures",
        help=(
            "Directory where finite-size comparison plots will be saved "
            "as PNG files (default: results/figures)."
        ),
    )
    return parser.parse_args()


def run_finite_size_sweeps(
    lattice_sizes: List[int],
    coupling_constant: float,
    temperature_min: float,
    temperature_max: float,
    number_of_temperatures: int,
    number_of_sweeps: int,
    burn_in_sweeps: int,
    sampling_interval: int,
    random_seed: int | None,
) -> Tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Run temperature sweeps for multiple lattice sizes.

    Args:
        lattice_sizes (List[int]): List of lattice sizes L.
        coupling_constant (float): Coupling constant J.
        temperature_min (float): Minimum temperature.
        temperature_max (float): Maximum temperature.
        number_of_temperatures (int): Number of temperature points.
        number_of_sweeps (int): Number of sweeps per temperature.
        burn_in_sweeps (int): Burn-in sweeps per temperature.
        sampling_interval (int): Sampling interval in sweeps.
        random_seed (int | None): Base random seed.

    Returns:
        Tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
            - Temperatures (shared for all sizes).
            - Dictionary mapping L to C_v(T) arrays.
            - Dictionary mapping L to chi(T) arrays.
    """
    Cv_by_size: dict[int, np.ndarray] = {}
    chi_by_size: dict[int, np.ndarray] = {}

    temperatures_reference: np.ndarray | None = None

    for index, lattice_size in enumerate(lattice_sizes):
        if random_seed is None:
            seed_for_size = None
        else:
            seed_for_size = random_seed + 100 * index

        (
            temperatures,
            magnetisation_means,
            specific_heats,
            susceptibilities,
        ) = run_temperature_sweep(
            lattice_size=lattice_size,
            coupling_constant=coupling_constant,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            number_of_temperatures=number_of_temperatures,
            number_of_sweeps=number_of_sweeps,
            burn_in_sweeps=burn_in_sweeps,
            sampling_interval=sampling_interval,
            random_seed=seed_for_size,
        )

        if temperatures_reference is None:
            temperatures_reference = temperatures
        else:
            if not np.allclose(temperatures_reference, temperatures):
                raise ValueError("Temperature grids do not match between sizes.")

        Cv_by_size[lattice_size] = specific_heats
        chi_by_size[lattice_size] = susceptibilities

        print(
            f"Completed sweep for L = {lattice_size}, "
            f"max C_v = {np.max(specific_heats):.3f}, "
            f"max chi = {np.max(susceptibilities):.3f}"
        )

    if temperatures_reference is None:
        raise RuntimeError("No lattice sizes were provided.")

    return temperatures_reference, Cv_by_size, chi_by_size


def make_finite_size_plots(
    temperatures: np.ndarray,
    Cv_by_size: dict[int, np.ndarray],
    chi_by_size: dict[int, np.ndarray],
    output_directory: str,
) -> None:
    """Create and save finite-size comparison plots.

    Args:
        temperatures (np.ndarray): Array of temperature values.
        Cv_by_size (dict[int, np.ndarray]): Specific heat curves by lattice size.
        chi_by_size (dict[int, np.ndarray]): Susceptibility curves by lattice size.
        output_directory (str): Directory where PNG files will be saved.
    """
    output_dir_path = Path(output_directory)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Specific heat comparison
    plt.figure()
    for lattice_size, Cv_values in sorted(Cv_by_size.items()):
        plt.plot(temperatures, Cv_values, marker="o", label=f"L = {lattice_size}")
    plt.xlabel("Temperature T")
    plt.ylabel("Specific heat per site C_v")
    plt.title("2D Ising model: finite-size behaviour of specific heat")
    plt.legend()
    Cv_path = output_dir_path / "ising2d_finite_size_specific_heat.png"
    plt.savefig(Cv_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Susceptibility comparison
    plt.figure()
    for lattice_size, chi_values in sorted(chi_by_size.items()):
        plt.plot(temperatures, chi_values, marker="o", label=f"L = {lattice_size}")
    plt.xlabel("Temperature T")
    plt.ylabel("Susceptibility per site chi")
    plt.title("2D Ising model: finite-size behaviour of susceptibility")
    plt.legend()
    chi_path = output_dir_path / "ising2d_finite_size_susceptibility.png"
    plt.savefig(chi_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved finite-size comparison plots to {output_dir_path}")


def main() -> None:
    """Entry point for the 2D Ising finite-size study."""
    arguments = parse_arguments()

    (
        temperatures,
        Cv_by_size,
        chi_by_size,
    ) = run_finite_size_sweeps(
        lattice_sizes=arguments.lattice_sizes,
        coupling_constant=arguments.coupling_constant,
        temperature_min=arguments.temperature_min,
        temperature_max=arguments.temperature_max,
        number_of_temperatures=arguments.number_of_temperatures,
        number_of_sweeps=arguments.number_of_sweeps,
        burn_in_sweeps=arguments.burn_in_sweeps,
        sampling_interval=arguments.sampling_interval,
        random_seed=arguments.random_seed,
    )

    make_finite_size_plots(
        temperatures=temperatures,
        Cv_by_size=Cv_by_size,
        chi_by_size=chi_by_size,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
