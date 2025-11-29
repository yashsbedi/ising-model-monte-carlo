"""
ising1d.py

Module for the 1D Ising model. Provides a class for initialising the spin
configuration, computing energies, and measuring magnetisation.

All code uses British English spelling and ASCII-only characters.
"""

from __future__ import annotations

import numpy as np


class Ising1DModel:
    """One-dimensional Ising model with periodic boundary conditions.

    Attributes:
        number_of_spins (int): Number of spins in the chain.
        coupling_constant (float): Interaction strength J between neighbours.
        spins (np.ndarray): Array of spin values, each +1 or -1.
    """

    def __init__(self, number_of_spins: int, coupling_constant: float = 1.0):
        """Initialise the 1D Ising model.

        Args:
            number_of_spins (int): Number of spins in the chain.
            coupling_constant (float): Nearest-neighbour coupling constant J.

        Raises:
            ValueError: If number_of_spins is not positive.
        """
        if number_of_spins <= 0:
            raise ValueError("number_of_spins must be positive.")

        self.number_of_spins = number_of_spins
        self.coupling_constant = coupling_constant
        self.spins = np.random.choice([-1, 1], size=number_of_spins)

    def compute_total_energy(self) -> float:
        """Compute the total energy of the current spin configuration.

        Returns:
            float: Total energy of the system.
        """
        shifted_spins = np.roll(self.spins, -1)
        total_energy = -self.coupling_constant * np.sum(self.spins * shifted_spins)
        return float(total_energy)

    def compute_magnetisation(self) -> float:
        """Compute the magnetisation per spin.

        Returns:
            float: Mean spin value.
        """
        return float(np.mean(self.spins))
