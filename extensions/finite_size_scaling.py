"""
finite_size_scaling.py

Finite-size scaling analysis for the 2D ferromagnetic Ising model.

This extension runs temperature sweeps for several lattice sizes,
extracts the peak values of the specific heat and susceptibility, and
performs log-log fits to estimate the scaling exponents alpha / nu and
gamma / nu. It also examines the shift of the pseudo-critical
temperature with system size.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.ising2d_finite_size_study import run_finite_size_sweeps


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the finite-size scaling analysis.

    Returns:
        argparse.Namespace: Parsed arguments, including the list of lattice
        sizes and simulation parameters.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Perform finite-size scaling analysis of the 2D ferromagnetic "
            "Ising model using specific heat and susceptibility peaks."
        )
    )
    parser.add_argument(
        "--lattice_sizes",
        type=int,
        nargs="+",
        default=[8, 16, 24, 32],
        help="List of lattice sizes L to study (default: 8 16 24 32).",
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
        default=1.5,
        help="Minimum temperature in the sweep (default: 1.5).",
    )
    parser.add_argument(
        "--temperature_max",
        type=float,
        default=3.5,
        help="Maximum temperature in the sweep (default: 3.5).",
    )
    parser.add_argument(
        "--number_of_temperatures",
        type=int,
        default=24,
        help="Number of temperature points in each sweep (default: 24).",
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
        default=987,
        help="Base random seed for reproducibility (default: 987).",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="results/figures",
        help=(
            "Directory where scaling plots will be saved as PNG files "
            "(default: results/figures)."
        ),
    )
    return parser.parse_args()


def extract_peak_observables(
    temperatures: np.ndarray,
    Cv_by_size: Dict[int, np.ndarray],
    chi_by_size: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract peak values and pseudo-critical temperatures.

    Args:
        temperatures (np.ndarray): Shared temperature grid.
        Cv_by_size (dict[int, np.ndarray]): Specific heat per site for each L.
        chi_by_size (dict[int, np.ndarray]): Susceptibility per site for each L.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Lattice sizes as a float array.
            - Peak specific heat values C_v^max(L).
            - Peak susceptibility values chi^max(L).
            - Pseudo-critical temperatures T_c^*(L), taken from the
              temperatures where the susceptibility peaks.
    """
    lattice_sizes = np.array(sorted(Cv_by_size.keys()), dtype=float)
    Cv_max_values = np.empty_like(lattice_sizes)
    chi_max_values = np.empty_like(lattice_sizes)
    Tc_star_values = np.empty_like(lattice_sizes)

    for index, L_value in enumerate(lattice_sizes):
        Cv_values = Cv_by_size[int(L_value)]
        chi_values = chi_by_size[int(L_value)]

        Cv_max_values[index] = np.max(Cv_values)
        chi_max_values[index] = np.max(chi_values)

        peak_index = int(np.argmax(chi_values))
        Tc_star_values[index] = temperatures[peak_index]

    return lattice_sizes, Cv_max_values, chi_max_values, Tc_star_values


def fit_power_law(
    lattice_sizes: np.ndarray,
    peak_values: np.ndarray,
) -> Tuple[float, float]:
    """Fit a power law peak(L) ~ L^{exponent} using log-log regression.

    Args:
        lattice_sizes (np.ndarray): Lattice sizes L.
        peak_values (np.ndarray): Peak observable values at each L.

    Returns:
        Tuple[float, float]: Estimated exponent and its intercept term in
        the log-log fit.
    """
    log_L = np.log(lattice_sizes)
    log_peak = np.log(peak_values)

    coefficients = np.polyfit(log_L, log_peak, deg=1)
    exponent = coefficients[0]
    intercept = coefficients[1]
    return float(exponent), float(intercept)


def make_scaling_plots(
    lattice_sizes: np.ndarray,
    Cv_max_values: np.ndarray,
    chi_max_values: np.ndarray,
    Tc_star_values: np.ndarray,
    Cv_exponent: float,
    chi_exponent: float,
    output_directory: str,
) -> None:
    """Create and save finite-size scaling plots.

    Args:
        lattice_sizes (np.ndarray): Lattice sizes L.
        Cv_max_values (np.ndarray): Peak specific heat values.
        chi_max_values (np.ndarray): Peak susceptibility values.
        Tc_star_values (np.ndarray): Pseudo-critical temperatures.
        Cv_exponent (float): Estimated alpha / nu from C_v peaks.
        chi_exponent (float): Estimated gamma / nu from chi peaks.
        output_directory (str): Directory where PNG files will be saved.
    """
    output_dir_path = Path(output_directory)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    log_L = np.log(lattice_sizes)

    # Specific heat scaling: log C_v^max vs log L
    log_Cv_max = np.log(Cv_max_values)
    Cv_fit_line = Cv_exponent * log_L + (
        log_Cv_max.mean() - Cv_exponent * log_L.mean()
    )

    plt.figure()
    plt.plot(log_L, log_Cv_max, marker="o", linestyle="none", label="Data")
    plt.plot(
        log_L,
        Cv_fit_line,
        label=f"Fit, slope = alpha / nu = {Cv_exponent:.3f}",
    )
    plt.xlabel("log L")
    plt.ylabel("log C_v^max(L)")
    plt.title("Finite-size scaling of specific heat peaks")
    plt.legend()
    Cv_path = output_dir_path / "ising2d_scaling_Cv_peaks.png"
    plt.savefig(Cv_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Susceptibility scaling: log chi^max vs log L
    log_chi_max = np.log(chi_max_values)
    chi_fit_line = chi_exponent * log_L + (
        log_chi_max.mean() - chi_exponent * log_L.mean()
    )

    plt.figure()
    plt.plot(log_L, log_chi_max, marker="o", linestyle="none", label="Data")
    plt.plot(
        log_L,
        chi_fit_line,
        label=f"Fit, slope = gamma / nu = {chi_exponent:.3f}",
    )
    plt.xlabel("log L")
    plt.ylabel("log chi^max(L)")
    plt.title("Finite-size scaling of susceptibility peaks")
    plt.legend()
    chi_path = output_dir_path / "ising2d_scaling_chi_peaks.png"
    plt.savefig(chi_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Pseudo-critical temperature shift: Tc*(L) vs 1 / L
    inverse_L = 1.0 / lattice_sizes

    plt.figure()
    plt.plot(inverse_L, Tc_star_values, marker="o")
    plt.xlabel("1 / L")
    plt.ylabel("Pseudo-critical temperature T_c^*(L)")
    plt.title("Shift of pseudo-critical temperature with system size")
    Tc_path = output_dir_path / "ising2d_scaling_Tc_shift.png"
    plt.savefig(Tc_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved finite-size scaling plots to {output_dir_path}")


def main() -> None:
    """Entry point for the finite-size scaling extension."""
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

    (
        lattice_sizes,
        Cv_max_values,
        chi_max_values,
        Tc_star_values,
    ) = extract_peak_observables(
        temperatures=temperatures,
        Cv_by_size=Cv_by_size,
        chi_by_size=chi_by_size,
    )

    Cv_exponent, _ = fit_power_law(
        lattice_sizes=lattice_sizes,
        peak_values=Cv_max_values,
    )
    chi_exponent, _ = fit_power_law(
        lattice_sizes=lattice_sizes,
        peak_values=chi_max_values,
    )

    print("Finite-size scaling estimates:")
    print(f"  alpha / nu from C_v peaks  ~= {Cv_exponent:.3f}")
    print(f"  gamma / nu from chi peaks ~= {chi_exponent:.3f}")

    make_scaling_plots(
        lattice_sizes=lattice_sizes,
        Cv_max_values=Cv_max_values,
        chi_max_values=chi_max_values,
        Tc_star_values=Tc_star_values,
        Cv_exponent=Cv_exponent,
        chi_exponent=chi_exponent,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
