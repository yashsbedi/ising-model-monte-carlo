"""
parallel_tempering.py

Parallel tempering (replica exchange) for the 2D Ising model.

This extension runs several replicas of the 2D Ising model at different
temperatures and performs periodic swap moves between neighbouring
temperatures. It records the energy and magnetisation time series for
each replica and produces a simple diagnostic plot for the replica
closest to the critical temperature.

The goal is to demonstrate improved sampling near the phase transition
and to provide data for further analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.ising2d import Ising2DModel
from src.metropolis import metropolis_single_flip_2d


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the parallel tempering script.

    Returns:
        argparse.Namespace: Parsed arguments including the lattice size,
        temperature ladder, simulation length, and output options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run parallel tempering (replica exchange) for the 2D Ising model "
            "and record energy and magnetisation time series."
        )
    )
    parser.add_argument(
        "--lattice_size",
        type=int,
        default=24,
        help="Linear lattice size L for an LxL system (default: 24).",
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
        default=1.8,
        help=(
            "Minimum temperature in the parallel tempering ladder "
            "(default: 1.8)."
        ),
    )
    parser.add_argument(
        "--temperature_max",
        type=float,
        default=3.0,
        help=(
            "Maximum temperature in the parallel tempering ladder "
            "(default: 3.0)."
        ),
    )
    parser.add_argument(
        "--number_of_replicas",
        type=int,
        default=8,
        help="Number of replicas (temperatures) in the ladder (default: 8).",
    )
    parser.add_argument(
        "--number_of_sweeps",
        type=int,
        default=40_000,
        help="Total number of Monte Carlo sweeps per replica (default: 40000).",
    )
    parser.add_argument(
        "--burn_in_sweeps",
        type=int,
        default=10_000,
        help="Number of initial sweeps discarded as burn-in (default: 10000).",
    )
    parser.add_argument(
        "--sampling_interval",
        type=int,
        default=50,
        help=(
            "Interval in sweeps between stored measurements "
            "(default: 50)."
        ),
    )
    parser.add_argument(
        "--swap_interval",
        type=int,
        default=10,
        help=(
            "Interval in sweeps between attempted replica-exchange moves "
            "(default: 10)."
        ),
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
        default="results",
        help=(
            "Base directory where raw data and figures will be saved "
            "(default: results). Raw data are stored under results/raw and "
            "figures under results/figures."
        ),
    )
    return parser.parse_args()


def build_temperature_ladder(
    temperature_min: float,
    temperature_max: float,
    number_of_replicas: int,
) -> np.ndarray:
    """Construct the temperature ladder for parallel tempering.

    The temperatures are spaced linearly between the minimum and maximum.

    Args:
        temperature_min (float): Minimum temperature.
        temperature_max (float): Maximum temperature.
        number_of_replicas (int): Number of replicas in the ladder.

    Returns:
        np.ndarray: Array of temperatures of shape (number_of_replicas,).
    """
    if number_of_replicas < 2:
        raise ValueError("Parallel tempering requires at least two replicas.")
    temperatures = np.linspace(
        temperature_min,
        temperature_max,
        number_of_replicas,
        dtype=float,
    )
    return temperatures


def compute_energy_and_magnetisation(
    spins: np.ndarray,
    coupling_constant: float,
) -> Tuple[float, float]:
    """Compute total energy and magnetisation for a 2D Ising configuration.

    The Hamiltonian is taken as
        E = -J * sum_{i,j} s_{i,j} (s_{i+1,j} + s_{i,j+1})
    with periodic boundary conditions, so each nearest neighbour bond is
    counted exactly once.

    Args:
        spins (np.ndarray): Spin configuration array of shape (L, L) with
            entries +/- 1.
        coupling_constant (float): Nearest neighbour coupling constant J.

    Returns:
        Tuple[float, float]: Total energy E and total magnetisation M.
    """
    # Interaction with right neighbours and down neighbours (periodic).
    shifted_right = np.roll(spins, shift=-1, axis=1)
    shifted_down = np.roll(spins, shift=-1, axis=0)
    interaction_sum = np.sum(spins * (shifted_right + shifted_down))
    energy = -coupling_constant * float(interaction_sum)
    magnetisation = float(np.sum(spins))
    return energy, magnetisation


def initialise_replicas(
    lattice_size: int,
    coupling_constant: float,
    temperatures: np.ndarray,
    random_seed: int | None,
) -> Tuple[List[Ising2DModel], np.ndarray, np.ndarray]:
    """Initialise replicas for parallel tempering.

    Args:
        lattice_size (int): Linear lattice size L.
        coupling_constant (float): Coupling constant J.
        temperatures (np.ndarray): Array of temperatures for each replica.
        random_seed (int | None): Optional random seed.

    Returns:
        Tuple[List[Ising2DModel], np.ndarray, np.ndarray]:
            - List of Ising2DModel instances, one per replica.
            - Array of energies E for each replica.
            - Array of magnetisations M for each replica.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    number_of_replicas = int(len(temperatures))
    models: List[Ising2DModel] = []
    energies = np.empty(number_of_replicas, dtype=float)
    magnetisations = np.empty(number_of_replicas, dtype=float)

    for index in range(number_of_replicas):
        model = Ising2DModel(
            lattice_size=lattice_size,
            coupling_constant=coupling_constant,
        )
        energy, magnetisation = compute_energy_and_magnetisation(
            spins=model.spins,
            coupling_constant=coupling_constant,
        )
        models.append(model)
        energies[index] = energy
        magnetisations[index] = magnetisation

    return models, energies, magnetisations


def attempt_swaps(
    models: List[Ising2DModel],
    energies: np.ndarray,
    betas: np.ndarray,
    coupling_constant: float,
    even_step: bool,
) -> Tuple[int, int]:
    """Attempt replica-exchange swaps between neighbouring temperatures.

    Swaps are proposed between pairs of neighbouring replicas. On an
    "even" step we attempt (0, 1), (2, 3), ... and on an "odd" step we
    attempt (1, 2), (3, 4), ... This helps the configurations diffuse
    across the full ladder.

    The acceptance probability for exchanging configurations between
    replicas i and j with inverse temperatures beta_i and beta_j is

        p_swap = min(1, exp[(beta_i - beta_j) (E_j - E_i)]).

    Args:
        models (List[Ising2DModel]): List of replicas.
        energies (np.ndarray): Current energies E_i for each replica.
        betas (np.ndarray): Inverse temperatures 1 / T_i for each replica.
        coupling_constant (float): Coupling constant J (not used here but
            kept for a consistent interface).
        even_step (bool): If True, attempt swaps (0,1), (2,3), ...;
            otherwise attempt (1,2), (3,4), ...

    Returns:
        Tuple[int, int]: Number of accepted swaps and number of attempted
        swaps in this call.
    """
    del coupling_constant  # Unused but kept for future extensions.

    number_of_replicas = len(models)
    start_index = 0 if even_step else 1

    accepted = 0
    attempted = 0

    for index in range(start_index, number_of_replicas - 1, 2):
        j_index = index + 1
        energy_i = energies[index]
        energy_j = energies[j_index]
        beta_i = betas[index]
        beta_j = betas[j_index]

        delta = (beta_i - beta_j) * (energy_j - energy_i)
        attempted += 1

        if delta >= 0.0 or np.random.rand() < np.exp(delta):
            # Accept swap: exchange the configurations and their energies.
            spins_i = models[index].spins.copy()
            models[index].spins[:, :] = models[j_index].spins
            models[j_index].spins[:, :] = spins_i

            energies[index], energies[j_index] = energies[j_index], energies[index]
            accepted += 1

    return accepted, attempted


def run_parallel_tempering(
    lattice_size: int,
    coupling_constant: float,
    temperatures: np.ndarray,
    number_of_sweeps: int,
    burn_in_sweeps: int,
    sampling_interval: int,
    swap_interval: int,
    random_seed: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the parallel tempering simulation.

    Args:
        lattice_size (int): Linear lattice size L.
        coupling_constant (float): Coupling constant J.
        temperatures (np.ndarray): Temperature ladder.
        number_of_sweeps (int): Total number of sweeps per replica.
        burn_in_sweeps (int): Number of sweeps discarded as burn-in.
        sampling_interval (int): Interval in sweeps between measurements.
        swap_interval (int): Interval in sweeps between swap attempts.
        random_seed (int | None): Optional random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Temperatures (copy of the input ladder).
            - Energy per site samples of shape (R, S) where R is the number
              of replicas and S is the number of stored samples.
            - Magnetisation per site samples of shape (R, S).
    """
    temperatures = np.asarray(temperatures, dtype=float)
    betas = 1.0 / temperatures

    (
        models,
        energies,
        magnetisations,
    ) = initialise_replicas(
        lattice_size=lattice_size,
        coupling_constant=coupling_constant,
        temperatures=temperatures,
        random_seed=random_seed,
    )

    number_of_replicas = len(models)
    if burn_in_sweeps >= number_of_sweeps:
        raise ValueError("Burn-in sweeps must be smaller than total sweeps.")

    number_of_samples = max(
        0, (number_of_sweeps - burn_in_sweeps) // sampling_interval
    )

    energy_samples = np.empty(
        (number_of_replicas, number_of_samples),
        dtype=float,
    )
    magnetisation_samples = np.empty(
        (number_of_replicas, number_of_samples),
        dtype=float,
    )

    accepted_swaps_total = 0
    attempted_swaps_total = 0
    even_step = True
    sample_index = 0

    number_of_sites = float(lattice_size * lattice_size)

    for sweep_index in range(number_of_sweeps):
        # Within-replica Metropolis updates.
        for replica_index in range(number_of_replicas):
            metropolis_single_flip_2d(
                models[replica_index],
                inverse_temperature=betas[replica_index],
            )
            energy, magnetisation = compute_energy_and_magnetisation(
                spins=models[replica_index].spins,
                coupling_constant=coupling_constant,
            )
            energies[replica_index] = energy
            magnetisations[replica_index] = magnetisation

        # Parallel tempering swap attempts every swap_interval sweeps.
        if (sweep_index + 1) % swap_interval == 0:
            accepted, attempted = attempt_swaps(
                models=models,
                energies=energies,
                betas=betas,
                coupling_constant=coupling_constant,
                even_step=even_step,
            )
            accepted_swaps_total += accepted
            attempted_swaps_total += attempted
            even_step = not even_step

        # Record measurements after burn-in.
        if (
            sweep_index >= burn_in_sweeps
            and (sweep_index - burn_in_sweeps) % sampling_interval == 0
        ):
            for replica_index in range(number_of_replicas):
                energy_per_site = energies[replica_index] / number_of_sites
                magnetisation_per_site = (
                    magnetisations[replica_index] / number_of_sites
                )
                energy_samples[replica_index, sample_index] = energy_per_site
                magnetisation_samples[replica_index, sample_index] = (
                    magnetisation_per_site
                )
            sample_index += 1

    if attempted_swaps_total > 0:
        swap_acceptance = accepted_swaps_total / attempted_swaps_total
    else:
        swap_acceptance = 0.0

    print(
        "Parallel tempering summary:"
        f" replicas = {number_of_replicas},"
        f" sweeps = {number_of_sweeps},"
        f" samples = {number_of_samples}"
    )
    print(
        "Total swap attempts ="
        f" {attempted_swaps_total},"
        f" accepted = {accepted_swaps_total},"
        f" acceptance rate = {swap_acceptance:.3f}"
    )

    return temperatures.copy(), energy_samples, magnetisation_samples


def save_parallel_tempering_data(
    temperatures: np.ndarray,
    energy_samples: np.ndarray,
    magnetisation_samples: np.ndarray,
    lattice_size: int,
    output_directory: str,
) -> Path:
    """Save parallel tempering data to a compressed NumPy file.

    Args:
        temperatures (np.ndarray): Temperature ladder.
        energy_samples (np.ndarray): Energy per site samples (R, S).
        magnetisation_samples (np.ndarray): Magnetisation per site samples
            (R, S).
        lattice_size (int): Lattice size L.
        output_directory (str): Base output directory.

    Returns:
        Path: Path to the saved .npz file.
    """
    base_dir = Path(output_directory)
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    output_path = raw_dir / f"ising2d_parallel_tempering_L_{lattice_size}.npz"
    np.savez_compressed(
        output_path,
        temperatures=temperatures,
        energy_samples=energy_samples,
        magnetisation_samples=magnetisation_samples,
        lattice_size=lattice_size,
    )

    print(f"Saved parallel tempering data to {output_path}")
    return output_path


def make_energy_trace_plot(
    temperatures: np.ndarray,
    energy_samples: np.ndarray,
    lattice_size: int,
    output_directory: str,
) -> None:
    """Create a simple energy trace plot for a central replica.

    The plot shows the energy per site as a function of sample index for
    the replica whose temperature is closest to the middle of the ladder.
    This gives a quick visual check of sampling behaviour near the
    critical region.

    Args:
        temperatures (np.ndarray): Temperature ladder.
        energy_samples (np.ndarray): Energy per site samples (R, S).
        lattice_size (int): Lattice size L.
        output_directory (str): Base output directory for figures.
    """
    figures_dir = Path(output_directory) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    number_of_replicas, number_of_samples = energy_samples.shape
    if number_of_samples == 0:
        print("No samples available for plotting.")
        return

    central_index = number_of_replicas // 2
    central_temperature = temperatures[central_index]
    central_energies = energy_samples[central_index]

    sample_indices = np.arange(number_of_samples, dtype=int)

    plt.figure()
    plt.plot(sample_indices, central_energies, marker="o", linestyle="-")
    plt.xlabel("Sample index")
    plt.ylabel("Energy per site")
    plt.title(
        "Parallel tempering energy trace at T = "
        f"{central_temperature:.3f}, L = {lattice_size}"
    )

    output_path = (
        figures_dir
        / f"ising2d_parallel_tempering_energy_trace_L_{lattice_size}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved energy trace plot to {output_path}")


def main() -> None:
    """Entry point for the 2D Ising parallel tempering script."""
    arguments = parse_arguments()

    temperatures = build_temperature_ladder(
        temperature_min=arguments.temperature_min,
        temperature_max=arguments.temperature_max,
        number_of_replicas=arguments.number_of_replicas,
    )

    (
        temperatures,
        energy_samples,
        magnetisation_samples,
    ) = run_parallel_tempering(
        lattice_size=arguments.lattice_size,
        coupling_constant=arguments.coupling_constant,
        temperatures=temperatures,
        number_of_sweeps=arguments.number_of_sweeps,
        burn_in_sweeps=arguments.burn_in_sweeps,
        sampling_interval=arguments.sampling_interval,
        swap_interval=arguments.swap_interval,
        random_seed=arguments.random_seed,
    )

    save_parallel_tempering_data(
        temperatures=temperatures,
        energy_samples=energy_samples,
        magnetisation_samples=magnetisation_samples,
        lattice_size=arguments.lattice_size,
        output_directory=arguments.output_directory,
    )

    make_energy_trace_plot(
        temperatures=temperatures,
        energy_samples=energy_samples,
        lattice_size=arguments.lattice_size,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
