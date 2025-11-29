"""
utils.py

General utility functions for the Ising project, including random seeding and
helper routines that are shared between modules.
"""

from __future__ import annotations

import numpy as np


def set_random_seed(seed_value: int) -> None:
    """Set the NumPy random seed for reproducible experiments.

    Args:
        seed_value (int): Seed value to pass to NumPy.
    """
    np.random.seed(seed_value)
