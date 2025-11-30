"""
observables.py

Functions to compute thermodynamic observables from Monte Carlo samples,
including specific heat and susceptibility.

All inputs are assumed to be NumPy arrays of floats.
"""

from __future__ import annotations

import numpy as np


def compute_specific_heat(
    energy_samples: np.ndarray,
    temperature: float,
) -> float:
    """Compute the specific heat from energy fluctuations.

    The specific heat C_v is defined as

        C_v = ( <E^2> - <E>^2 ) / T^2,

    where E is the total energy of the system.

    Args:
        energy_samples (np.ndarray): Array of sampled total energies E.
        temperature (float): Simulation temperature T.

    Returns:
        float: Specific heat C_v for the whole system.
    """
    energies = np.asarray(energy_samples, dtype=float)
    mean_E = np.mean(energies)
    mean_E2 = np.mean(energies ** 2)
    variance_E = mean_E2 - mean_E * mean_E
    specific_heat = variance_E / (temperature * temperature)
    return float(specific_heat)


def compute_susceptibility(
    magnetisation_samples: np.ndarray,
    temperature: float,
    number_of_sites: int,
) -> float:
    """Compute the magnetic susceptibility per site.

    The input magnetisation_samples are assumed to be magnetisation per site

        m = M / N,

    where M is the total magnetisation and N is the number of sites.
    The susceptibility per site is then

        chi = N * ( <m^2> - <m>^2 ) / T.

    Args:
        magnetisation_samples (np.ndarray): Array of sampled magnetisations
            per site.
        temperature (float): Simulation temperature T.
        number_of_sites (int): Total number of lattice sites N.

    Returns:
        float: Susceptibility per site chi.
    """
    m = np.asarray(magnetisation_samples, dtype=float)
    mean_m = np.mean(m)
    mean_m2 = np.mean(m ** 2)
    variance_m = mean_m2 - mean_m * mean_m
    susceptibility = number_of_sites * variance_m / temperature
    return float(susceptibility)
