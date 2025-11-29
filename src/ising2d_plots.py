"""
ising2d_plots.py

Plot magnetisation and specific heat as functions of temperature for the
2D Ising model, using data produced by the temperature sweep script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the plotting script.

    Returns:
        argparse.Namespace: Parsed arguments containing the input file path
        and optional output directory.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot magnetisation and specific heat versus temperature for the "
            "2D Ising model using precomputed sweep data."
        )
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="results/raw/ising2d_temperature_sweep.npz",
        help=(
            "Path to the .npz file produced by ising2d_temperature_sweep "
            "(default: results/raw/ising2d_temperature_sweep.npz)."
        ),
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default="results/figures",
        help=(
            "Directory where plots will be saved as PNG files "
            "(default: results/figures)."
        ),
    )
    return parser.parse_args()


def load_sweep_data(input_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load temperature sweep data from a .npz file.

    Args:
        input_path (str): Path to the .npz file containing sweep data.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Temperatures.
            - Mean magnetisation values.
            - Specific heat values.
    """
    data = np.load(input_path)
    temperatures = data["temperatures"]
    magnetisation_means = data["magnetisation_means"]
    specific_heats = data["specific_heats"]
    return temperatures, magnetisation_means, specific_heats


def make_plots(
    temperatures: np.ndarray,
    magnetisation_means: np.ndarray,
    specific_heats: np.ndarray,
    output_directory: str,
) -> None:
    """Create and save plots of |M| and C_v as functions of temperature.

    Args:
        temperatures (np.ndarray): Array of temperature values.
        magnetisation_means (np.ndarray): Mean magnetisation values.
        specific_heats (np.ndarray): Specific heat values.
        output_directory (str): Directory where PNG files will be saved.
    """
    output_dir_path = Path(output_directory)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Plot magnetisation
    plt.figure()
    plt.plot(temperatures, magnetisation_means, marker="o")
    plt.xlabel("Temperature T")
    plt.ylabel("Mean |M| per site")
    plt.title("2D Ising model: magnetisation versus temperature")
    magnetisation_path = output_dir_path / "ising2d_magnetisation_vs_temperature.png"
    plt.savefig(magnetisation_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot specific heat
    plt.figure()
    plt.plot(temperatures, specific_heats, marker="o")
    plt.xlabel("Temperature T")
    plt.ylabel("Specific heat per site C_v")
    plt.title("2D Ising model: specific heat versus temperature")
    specific_heat_path = output_dir_path / "ising2d_specific_heat_vs_temperature.png"
    plt.savefig(specific_heat_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {output_dir_path}")


def main() -> None:
    """Entry point for generating plots from 2D Ising sweep data."""
    arguments = parse_arguments()
    temperatures, magnetisation_means, specific_heats = load_sweep_data(
        input_path=arguments.input_path
    )
    make_plots(
        temperatures=temperatures,
        magnetisation_means=magnetisation_means,
        specific_heats=specific_heats,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
