"""
ising2d_temperature_sweep.py

Run a 2D Ising model over a range of temperatures, using the Metropolis
algorithm, and record magnetisation, specific heat, and susceptibility
as functions of temperature.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .ising2d_simulation import run_ising2d_simulation
from .observables import compute_specific_heat, compute_susceptibility


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the 2D temperature sweep.

    Returns:
        argparse.Namespace: Parsed arguments controlling lattice size,
        coupling constant, temperature range, and simulation parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run a 2D Ising model over a range of temperatures and compute "
            "magnetisation, specific heat, and susceptibility."
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
        help=(
            "Nearest neighbour coupling constant J (default: 1.0). "
            "Use a negative value for the antiferromagnetic case."
        ),
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
        help="Number of temperature points in the sweep (default: 20).",
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
        help="Optional base seed. Different temperatures will use offsets.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/raw/ising2d_temperature_sweep.npz",
        help=(
            "Path to a .npz file where temperature-dependent observables will "
            "be stored (default: results/raw/ising2d_temperature_sweep.npz)."
        ),
    )
    return parser.parse_args()


def run_temperature_sweep(
    lattice_size: int,
    coupling_constant: float,
    temperature_min: float,
    temperature_max: float,
    number_of_temperatures: int,
    number_of_sweeps: int,
    burn_in_sweeps: int,
    sampling_interval: int,
    random_seed: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a temperature sweep and compute magnetisation, C_v and chi.

    Args:
        lattice_size (int): Linear lattice size L.
        coupling_constant (float): Coupling constant J.
        temperature_min (float): Minimum temperature.
        temperature_max (float): Maximum temperature.
        number_of_temperatures (int): Number of temperature points.
        number_of_sweeps (int): Number of sweeps per temperature.
        burn_in_sweeps (int): Burn-in sweeps per temperature.
        sampling_interval (int): Sampling interval in sweeps.
        random_seed (int | None): Base seed.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Temperatures.
            - Mean absolute magnetisation per site.
            - Specific heat per site.
            - Susceptibility per site.
    """
    temperatures = np.linspace(
        temperature_min, temperature_max, number_of_temperatures
    )
    magnetisation_means = np.empty(number_of_temperatures, dtype=float)
    specific_heats = np.empty(number_of_temperatures, dtype=float)
    susceptibilities = np.empty(number_of_temperatures, dtype=float)

    number_of_sites = lattice_size * lattice_size

    for index, temperature in enumerate(temperatures):
        if random_seed is None:
            seed_for_this_temperature = None
        else:
            seed_for_this_temperature = random_seed + index

        energy_samples, magnetisation_samples = run_ising2d_simulation(
            lattice_size=lattice_size,
            temperature=temperature,
            coupling_constant=coupling_constant,
            number_of_sweeps=number_of_sweeps,
            burn_in_sweeps=burn_in_sweeps,
            sampling_interval=sampling_interval,
            random_seed=seed_for_this_temperature,
        )

        # Mean absolute magnetisation per site
        magnetisation_means[index] = np.mean(np.abs(magnetisation_samples))

        # Specific heat per site
        Cv_total = compute_specific_heat(
            energy_samples=np.array(energy_samples),
            temperature=temperature,
        )
        specific_heats[index] = Cv_total / number_of_sites

        # Susceptibility per site
        susceptibilities[index] = compute_susceptibility(
            magnetisation_samples=np.array(magnetisation_samples),
            temperature=temperature,
            number_of_sites=number_of_sites,
        )

        print(
            f"T = {temperature:.3f}  "
            f"<|M|> = {magnetisation_means[index]:.4f}  "
            f"C_v = {specific_heats[index]:.4f}  "
            f"chi = {susceptibilities[index]:.4f}"
        )

    return temperatures, magnetisation_means, specific_heats, susceptibilities


def main() -> None:
    """Entry point for the 2D Ising temperature sweep."""
    arguments = parse_arguments()

    (
        temperatures,
        magnetisation_means,
        specific_heats,
        susceptibilities,
    ) = run_temperature_sweep(
        lattice_size=arguments.lattice_size,
        coupling_constant=arguments.coupling_constant,
        temperature_min=arguments.temperature_min,
        temperature_max=arguments.temperature_max,
        number_of_temperatures=arguments.number_of_temperatures,
        number_of_sweeps=arguments.number_of_sweeps,
        burn_in_sweeps=arguments.burn_in_sweeps,
        sampling_interval=arguments.sampling_interval,
        random_seed=arguments.random_seed,
    )

    output_file = Path(arguments.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        temperatures=temperatures,
        magnetisation_means=magnetisation_means,
        specific_heats=specific_heats,
        susceptibilities=susceptibilities,
        lattice_size=arguments.lattice_size,
        coupling_constant=arguments.coupling_constant,
    )

    print(f"Saved sweep data to {arguments.output_path}")


if __name__ == "__main__":
    main()
