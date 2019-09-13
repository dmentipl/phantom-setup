from typing import Union

import numpy as np

from . import constants


def keplerian_angular_velocity(
        radius: Union[float, np.ndarray],
        mass: Union[float, np.ndarray],
        gravitational_constant: float = None):
    """
    Keplerian angular velocity Omega.

    Parameters
    ----------
    radius
        The distance from the central object.
    mass
        The central object mass.
    gravitational_constant
        The gravitational constant in appropriate units.
    """
    if gravitational_constant is None:
        gravitational_constant = constants.gravitational_constant
    return np.sqrt(gravitational_constant * mass / radius)
