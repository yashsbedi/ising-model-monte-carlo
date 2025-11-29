"""
ising1d_simulation.py

Command-line interface for running a 1D Ising model simulation at a single
temperature using the Metropolis algorithm. The script performs an initial
equilibration (burn-in) period and then records samples of the energy and
magnetisation for later analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from .ising1d import Ising1DModel
from .metropolis import metropolis_single_flip_1d


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the 1D Ising simulation.

    Returns:
        argparse.Namespace: Parsed arguments controlling lattice size,
        temperature, number of sweeps, and output location.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run a 1D Ising model simulation using the Metropolis algorithm "
            "and record energy and magnetisation samples."
        )
    )
    parser.add_argument(
        "--number_of_spins",
        type=int,
        default=64,
        help="Number of spins in the 1D chain (default: 64).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Simulation temperature T (default: 1.0).",
    )
    parser.add_argument(
        "--coupling_constant",
        type=float,
        default=1.0,
        help="Nearest neighbour coupling constant J (default: 1.0).",
    )
    parser.add_argument(
        "--number_of_sweeps",
        type=int,
        default=10_000,
        help="Total number of Monte Carlo sweeps (default: 10000).",
    )
    parser.add_argument(
        "--burn_in_sweeps",
        type=int,
        default=2_000,
        help="Number of initial sweeps to discard as burn-in (default: 2000).",
    )
    parser.add_argument(
        "--sampling_interval",
        type=int,
        default=10,
        help=(
            "Record observables every 'sampling_interval' sweeps "
            "(default: 10)."
        ),
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
        default="results/raw/ising1d_samples.npz",
        help=(
            "Path to a .npz file where energy and magnetisation samples will "
            "be stored (default: results/raw/ising1d_samples.npz)."
        ),
    )
    return parser.parse_args()


def run_ising1d_simulation(
    number_of_spins: int,
    temperature: float,
    coupling_constant: float,
    number_of_sweeps: int,
    burn_in_sweeps: int,
    sampling_interval: int,
    random_seed: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a 1D Ising Monte Carlo simulation and collect samples.

    Args:
        number_of_spins (int): Number of spins in the chain.
        temperature (float): Simulation temperature.
        coupling_constant (float): Nearest neighbour coupling constant J.
        number_of_sweeps (int): Total number of Monte Carlo sweeps.
        burn_in_sweeps (int): Number of initial sweeps to discard.
        sampling_interval (int): Record observables every this many sweeps.
        random_seed (int | None): Optional integer seed.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Array of sampled energies.
            - Array of sampled magnetisations.
    """
    if random_seed is not None:
        random_number_generator = np.random.default_rng(random_seed)
    else:
        random_number_generator = np.random.default_rng()

    ising_model = Ising1DModel(
        number_of_spins=number_of_spins,
        coupling_constant=coupling_constant,
    )

    inverse_temperature = 1.0 / temperature

    number_of_samples = max(
        0, (number_of_sweeps - burn_in_sweeps) // sampling_interval
    )
    energy_samples = np.empty(number_of_samples, dtype=float)
    magnetisation_samples = np.empty(number_of_samples, dtype=float)

    sample_index = 0

    for sweep_index in range(number_of_sweeps):
        # One sweep = attempt to update each spin once on average.
        for _ in range(number_of_spins):
            metropolis_single_flip_1d(
                ising_model=ising_model,
                inverse_temperature=inverse_temperature,
                random_number_generator=random_number_generator,
            )

        if sweep_index >= burn_in_sweeps and (
            (sweep_index - burn_in_sweeps) % sampling_interval == 0
        ):
            energy_samples[sample_index] = ising_model.compute_total_energy()
            magnetisation_samples[sample_index] = (
                ising_model.compute_magnetisation()
            )
            sample_index += 1

    return energy_samples, magnetisation_samples


def main() -> None:
    """Entry point for running and saving a 1D Ising simulation."""
    arguments = parse_arguments()

    energy_samples, magnetisation_samples = run_ising1d_simulation(
        number_of_spins=arguments.number_of_spins,
        temperature=arguments.temperature,
        coupling_constant=arguments.coupling_constant,
        number_of_sweeps=arguments.number_of_sweeps,
        burn_in_sweeps=arguments.burn_in_sweeps,
        sampling_interval=arguments.sampling_interval,
        random_seed=arguments.random_seed,
    )

    output_file = Path(arguments.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        energy_samples=energy_samples,
        magnetisation_samples=magnetisation_samples,
        temperature=arguments.temperature,
        number_of_spins=arguments.number_of_spins,
    )

    print(f"Saved {len(energy_samples)} samples to {arguments.output_path}")
    print(
        f"Mean energy per spin        : "
        f"{np.mean(energy_samples) / arguments.number_of_spins:.6f}"
    )
    print(
        f"Mean magnetisation per spin : "
        f"{np.mean(magnetisation_samples):.6f}"
    )


if __name__ == "__main__":
    main()
