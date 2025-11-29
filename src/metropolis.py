"""
metropolis.py

Metropolis update steps for 1D and 2D Ising models. These routines implement
single spin-flip proposals with Boltzmann acceptance probabilities.
"""

from __future__ import annotations

import numpy as np

from .ising1d import Ising1DModel
from .ising2d import Ising2DModel


def metropolis_single_flip_1d(
    ising_model: Ising1DModel,
    inverse_temperature: float,
    random_number_generator: np.random.Generator | None = None,
) -> None:
    """Perform a single Metropolis spin-flip update for a 1D Ising model.

    Args:
        ising_model (Ising1DModel): The Ising model to update.
        inverse_temperature (float): Inverse temperature beta = 1.0 / T.
        random_number_generator (np.random.Generator | None): Optional RNG. If
            None, a default generator is used.
    """
    if random_number_generator is None:
        random_number_generator = np.random.default_rng()

    number_of_spins = ising_model.number_of_spins
    spin_index = random_number_generator.integers(low=0, high=number_of_spins)

    left_neighbour = ising_model.spins[(spin_index - 1) % number_of_spins]
    right_neighbour = ising_model.spins[(spin_index + 1) % number_of_spins]
    local_field = left_neighbour + right_neighbour

    delta_energy = 2.0 * ising_model.coupling_constant * ising_model.spins[spin_index] * local_field

    if delta_energy <= 0.0:
        ising_model.spins[spin_index] *= -1
    else:
        acceptance_probability = np.exp(-inverse_temperature * delta_energy)
        if random_number_generator.random() < acceptance_probability:
            ising_model.spins[spin_index] *= -1


def metropolis_single_flip_2d(
    ising_model: Ising2DModel,
    inverse_temperature: float,
    random_number_generator: np.random.Generator | None = None,
) -> None:
    """Perform a single Metropolis spin-flip update for a 2D Ising model.

    Args:
        ising_model (Ising2DModel): The Ising model to update.
        inverse_temperature (float): Inverse temperature beta = 1.0 / T.
        random_number_generator (np.random.Generator | None): Optional RNG. If
            None, a default generator is used.
    """
    if random_number_generator is None:
        random_number_generator = np.random.default_rng()

    lattice_size = ising_model.lattice_size
    row_index = random_number_generator.integers(low=0, high=lattice_size)
    column_index = random_number_generator.integers(low=0, high=lattice_size)

    spin_value = ising_model.spins[row_index, column_index]

    neighbour_sum = (
        ising_model.spins[(row_index - 1) % lattice_size, column_index]
        + ising_model.spins[(row_index + 1) % lattice_size, column_index]
        + ising_model.spins[row_index, (column_index - 1) % lattice_size]
        + ising_model.spins[row_index, (column_index + 1) % lattice_size]
    )

    delta_energy = 2.0 * ising_model.coupling_constant * spin_value * neighbour_sum

    if delta_energy <= 0.0:
        ising_model.spins[row_index, column_index] *= -1
    else:
        acceptance_probability = np.exp(-inverse_temperature * delta_energy)
        if random_number_generator.random() < acceptance_probability:
            ising_model.spins[row_index, column_index] *= -1
