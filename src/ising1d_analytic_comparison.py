"""
ising1d_analytic_comparison.py

Run a 1D Ising model over a range of temperatures and compare the Monte
Carlo estimate of the mean energy per spin with the exact analytic
transfer-matrix result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from .ising1d_simulation import run_ising1d_simulation


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the 1D Ising comparison script.

    Returns:
        argparse.Namespace: Parsed arguments controlling lattice size,
        coupling constant, temperature range, and simulation parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare 1D Ising Monte Carlo energies with the exact "
            "transfer-matrix result."
        )
    )
    parser.add_argument(
        "--number_of_spins",
        type=int,
        default=64,
        help="Number of spins in the 1D chain (default: 64).",
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
        help="Minimum temperature in the sweep (must be > 0, default: 0.01).",
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
        default=10_000,
        help="Total number of Monte Carlo sweeps per temperature (default: 10000).",
    )
    parser.add_argument(
        "--burn_in_sweeps",
        type=int,
        default=2_000,
        help="Number of initial sweeps discarded as burn in (default: 2000).",
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
        help="Optional base seed. Different temperatures use offsets.",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="results/figures",
        help=(
            "Directory where the comparison plot will be saved as a PNG "
            "(default: results/figures)."
        ),
    )
    return parser.parse_args()


def analytic_energy_per_spin(
    temperature_array: np.ndarray,
    coupling_constant: float,
) -> np.ndarray:
    """Compute the exact 1D Ising energy per spin.

    Uses the standard transfer-matrix result for zero external field:

        E / N = -J * tanh(J / T).

    Args:
        temperature_array (np.ndarray): Temperatures at which to evaluate
            the analytic result.
        coupling_constant (float): Coupling constant J.

    Returns:
        np.ndarray: Analytic mean energy per spin at each temperature.
    """
    beta_array = 1.0 / temperature_array
    return -coupling_constant * np.tanh(beta_array * coupling_constant)


def run_temperature_sweep_monte_carlo(
    number_of_spins: int,
    coupling_constant: float,
    temperature_min: float,
    temperature_max: float,
    number_of_temperatures: int,
    number_of_sweeps: int,
    burn_in_sweeps: int,
    sampling_interval: int,
    random_seed: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run 1D Ising simulations over a temperature range.

    Args:
        number_of_spins (int): Number of spins in the chain.
        coupling_constant (float): Coupling constant J.
        temperature_min (float): Minimum temperature in the sweep.
        temperature_max (float): Maximum temperature in the sweep.
        number_of_temperatures (int): Number of temperature points.
        number_of_sweeps (int): Number of sweeps per temperature.
        burn_in_sweeps (int): Burn in sweeps per temperature.
        sampling_interval (int): Sampling interval in sweeps.
        random_seed (int | None): Base random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Temperatures.
            - Monte Carlo mean energies per spin at each temperature.
    """
    if temperature_min <= 0.0:
        raise ValueError("temperature_min must be strictly positive.")

    temperatures = np.linspace(
        temperature_min, temperature_max, number_of_temperatures
    )
    mean_energies_per_spin = np.empty(number_of_temperatures, dtype=float)

    for index, temperature in enumerate(temperatures):
        if random_seed is None:
            seed_for_this_temperature = None
        else:
            seed_for_this_temperature = random_seed + index

        energy_samples, magnetisation_samples = run_ising1d_simulation(
            number_of_spins=number_of_spins,
            temperature=temperature,
            coupling_constant=coupling_constant,
            number_of_sweeps=number_of_sweeps,
            burn_in_sweeps=burn_in_sweeps,
            sampling_interval=sampling_interval,
            random_seed=seed_for_this_temperature,
        )

        mean_energies_per_spin[index] = np.mean(energy_samples) / float(
            number_of_spins
        )

        print(
            f"T = {temperature:.3f}  "
            f"<E>/N (MC) = {mean_energies_per_spin[index]:.4f}"
        )

    return temperatures, mean_energies_per_spin


def make_energy_comparison_plot(
    temperatures: np.ndarray,
    mean_energies_per_spin_mc: np.ndarray,
    coupling_constant: float,
    output_directory: str,
) -> None:
    """Create and save a plot comparing Monte Carlo and analytic energies.

    Args:
        temperatures (np.ndarray): Temperature values used in the sweep.
        mean_energies_per_spin_mc (np.ndarray): Monte Carlo energies per spin.
        coupling_constant (float): Coupling constant J.
        output_directory (str): Directory where the PNG will be saved.
    """
    output_dir_path = Path(output_directory)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    mean_energies_per_spin_exact = analytic_energy_per_spin(
        temperature_array=temperatures,
        coupling_constant=coupling_constant,
    )

    plt.figure()
    plt.plot(
        temperatures,
        mean_energies_per_spin_exact,
        label="Exact transfer matrix",
    )
    plt.plot(
        temperatures,
        mean_energies_per_spin_mc,
        marker="o",
        linestyle="none",
        label="Monte Carlo",
    )
    plt.xlabel("Temperature T")
    plt.ylabel("Mean energy per spin")
    plt.title("1D Ising model: energy per spin versus temperature")
    plt.legend()

    output_path = output_dir_path / "ising1d_energy_analytic_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot to {output_path}")


def main() -> None:
    """Entry point for the 1D Ising analytic comparison script."""
    arguments = parse_arguments()

    (
        temperatures,
        mean_energies_per_spin_mc,
    ) = run_temperature_sweep_monte_carlo(
        number_of_spins=arguments.number_of_spins,
        coupling_constant=arguments.coupling_constant,
        temperature_min=arguments.temperature_min,
        temperature_max=arguments.temperature_max,
        number_of_temperatures=arguments.number_of_temperatures,
        number_of_sweeps=arguments.number_of_sweeps,
        burn_in_sweeps=arguments.burn_in_sweeps,
        sampling_interval=arguments.sampling_interval,
        random_seed=arguments.random_seed,
    )

    make_energy_comparison_plot(
        temperatures=temperatures,
        mean_energies_per_spin_mc=mean_energies_per_spin_mc,
        coupling_constant=arguments.coupling_constant,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
