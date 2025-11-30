"""
ising2d.py

Module for the 2D square-lattice Ising model. Provides a class for
initialising the lattice, computing energies, and measuring magnetisation.

For the antiferromagnetic case, the coupling constant should be negative.
"""

from __future__ import annotations

import numpy as np


class Ising2DModel:
    """Two-dimensional Ising model with periodic boundary conditions.

    Attributes:
        lattice_size (int): Linear size of the square lattice.
        coupling_constant (float): Nearest-neighbour coupling constant J.
        spins (np.ndarray): Two-dimensional array of spins (+1 or -1).
    """

    def __init__(self, lattice_size: int, coupling_constant: float = 1.0):
        """Initialise the 2D Ising model.

        Args:
            lattice_size (int): Linear lattice dimension (lattice_size x lattice_size).
            coupling_constant (float): Nearest-neighbour coupling constant J.
                Use J < 0 for the antiferromagnetic case.

        Raises:
            ValueError: If lattice_size is not positive.
        """
        if lattice_size <= 0:
            raise ValueError("lattice_size must be positive.")

        self.lattice_size = lattice_size
        self.coupling_constant = coupling_constant
        self.spins = np.random.choice([-1, 1], size=(lattice_size, lattice_size))

    def compute_total_energy(self) -> float:
        """Compute the total energy of the 2D lattice with periodic boundaries.

        Returns:
            float: Total energy of the system.
        """
        # Neighbours: up and right (others accounted for by periodicity)
        spins_up = np.roll(self.spins, shift=-1, axis=0)
        spins_right = np.roll(self.spins, shift=-1, axis=1)
        interaction_sum = np.sum(self.spins * (spins_up + spins_right))
        total_energy = -self.coupling_constant * interaction_sum
        return float(total_energy)

    def compute_magnetisation(self) -> float:
        """Compute the magnetisation per spin.

        Returns:
            float: Mean spin value.
        """
        return float(np.mean(self.spins))


    def compute_staggered_magnetisation(self) -> float:
        """Compute the staggered magnetisation per site.

        The staggered magnetisation is defined as

            m_s = (1 / N) * sum_{i,j} (-1)^{i + j} s_{i,j},

        which is the appropriate order parameter for the antiferromagnetic
        case, where neighbouring spins prefer to be anti-aligned.

        Returns:
            float: Staggered magnetisation per site.
        """
        lattice_size = self.lattice_size
        # Create a checkerboard pattern of +1 and -1 factors
        row_indices, column_indices = np.indices((lattice_size, lattice_size))
        checkerboard = (-1) ** (row_indices + column_indices)
        staggered = np.mean(self.spins * checkerboard)
        return float(staggered)
