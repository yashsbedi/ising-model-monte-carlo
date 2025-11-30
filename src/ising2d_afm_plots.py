"""
ising2d_afm_plots.py

Plot staggered magnetisation, specific heat, and susceptibility as
functions of temperature for the 2D antiferromagnetic Ising model,
using data produced by ising2d_afm_temperature_sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the AFM plotting script.

    Returns:
        argparse.Namespace: Parsed arguments containing the input file path
        and output directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot staggered magnetisation, specific heat, and susceptibility "
            "versus temperature for the 2D antiferromagnetic Ising model."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="results/raw/ising2d_afm_temperature_sweep.npz",
        help=(
            "Path to the .npz file produced by ising2d_afm_temperature_sweep "
            "(default: results/raw/ising2d_afm_temperature_sweep.npz)."
        ),
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="results/figures",
        help=(
            "Directory where AFM plots will be saved as PNG files "
            "(default: results/figures)."
        ),
    )
    return parser.parse_args()


def load_afm_sweep_data(
    input_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load AFM temperature sweep data from a .npz file.

    Args:
        input_path (str): Path to the .npz file containing AFM sweep data.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Temperatures.
            - Mean staggered magnetisation values.
            - Specific heat values.
            - Susceptibility values.
    """
    data = np.load(input_path)
    temperatures = data["temperatures"]
    staggered_magnetisation_means = data["staggered_magnetisation_means"]
    specific_heats = data["specific_heats"]
    susceptibilities = data["susceptibilities"]
    return (
        temperatures,
        staggered_magnetisation_means,
        specific_heats,
        susceptibilities,
    )


def make_afm_plots(
    temperatures: np.ndarray,
    staggered_magnetisation_means: np.ndarray,
    specific_heats: np.ndarray,
    susceptibilities: np.ndarray,
    output_directory: str,
) -> None:
    """Create and save AFM plots as PNG files.

    Args:
        temperatures (np.ndarray): Array of temperature values.
        staggered_magnetisation_means (np.ndarray): Mean staggered
            magnetisation per site.
        specific_heats (np.ndarray): Specific heat per site values.
        susceptibilities (np.ndarray): Susceptibility per site values.
        output_directory (str): Directory where PNG files will be saved.
    """
    output_dir_path = Path(output_directory)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Staggered magnetisation
    plt.figure()
    plt.plot(temperatures, staggered_magnetisation_means, marker="o")
    plt.xlabel("Temperature T")
    plt.ylabel("Mean |M_s| per site")
    plt.title("2D AFM Ising model: staggered magnetisation versus temperature")
    ms_path = (
        output_dir_path
        / "ising2d_afm_staggered_magnetisation_vs_temperature.png"
    )
    plt.savefig(ms_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Specific heat
    plt.figure()
    plt.plot(temperatures, specific_heats, marker="o")
    plt.xlabel("Temperature T")
    plt.ylabel("Specific heat per site C_v")
    plt.title("2D AFM Ising model: specific heat versus temperature")
    cv_path = output_dir_path / "ising2d_afm_specific_heat_vs_temperature.png"
    plt.savefig(cv_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Susceptibility based on staggered magnetisation
    plt.figure()
    plt.plot(temperatures, susceptibilities, marker="o")
    plt.xlabel("Temperature T")
    plt.ylabel("Susceptibility per site chi_s")
    plt.title("2D AFM Ising model: susceptibility versus temperature")
    chi_path = output_dir_path / "ising2d_afm_susceptibility_vs_temperature.png"
    plt.savefig(chi_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved AFM plots to {output_dir_path}")
    


def main() -> None:
    """Entry point for generating AFM plots from sweep data."""
    arguments = parse_arguments()
    (
        temperatures,
        staggered_magnetisation_means,
        specific_heats,
        susceptibilities,
    ) = load_afm_sweep_data(input_path=arguments.input_path)
    make_afm_plots(
        temperatures=temperatures,
        staggered_magnetisation_means=staggered_magnetisation_means,
        specific_heats=specific_heats,
        susceptibilities=susceptibilities,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
