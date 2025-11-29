"""
observables.py

Functions to compute thermodynamic observables from Monte Carlo samples,
including specific heat, susceptibility, and basic correlation estimates.
"""

from __future__ import annotations

import numpy as np


def compute_specific_heat(
    energy_samples: np.ndarray,
    temperature: float,
) -> float:
    """Compute the specific heat per spin from energy fluctuations.

    Args:
        energy_samples (np.ndarray): Array of sampled total energies.
        temperature (float): Simulation temperature.

    Returns:
        float: Specific heat per spin.
    """
    energy_mean = np.mean(energy_samples)
    energy_squared_mean = np.mean(energy_samples ** 2)
    variance_energy = energy_squared_mean - energy_mean ** 2
    specific_heat = variance_energy / (temperature ** 2)
    return float(specific_heat)


def compute_susceptibility(
    magnetisation_samples: np.ndarray,
    temperature: float,
) -> float:
    """Compute the magnetic susceptibility per spin.

    Args:
        magnetisation_samples (np.ndarray): Array of sampled magnetisations.
        temperature (float): Simulation temperature.

    Returns:
        float: Magnetic susceptibility per spin.
    """
    magnetisation_mean = np.mean(magnetisation_samples)
    magnetisation_squared_mean = np.mean(magnetisation_samples ** 2)
    variance_magnetisation = magnetisation_squared_mean - magnetisation_mean ** 2
    susceptibility = variance_magnetisation / temperature
    return float(susceptibility)
