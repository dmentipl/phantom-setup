import functools
from typing import Union

import numpy as np

from . import constants


def keplerian_angular_velocity(
    radius: Union[float, np.ndarray],
    mass: Union[float, np.ndarray],
    gravitational_constant: float = None,
):
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
    return np.sqrt(gravitational_constant * mass / radius ** 3)


def add_gap(radius_planet: float, gap_width: float):
    """
    Decorator to add a gap in a density distribution.

    The gap is a step function. I.e. the density is zero within the gap.

    Parameters
    ----------
    radius_planet
        The planet radius.
    gap_width
        The gap width centered on the planet.

    Returns
    -------
    callable
        The density distribution with the added gap.
    """

    def wrapper_outer(distribution):
        @functools.wraps(distribution)
        def wrapper_inner(radius, *args):

            result = distribution(radius, *args)
            gap_radii = np.logical_and(
                radius_planet - 0.5 * gap_width < radius,
                radius < radius_planet + 0.5 * gap_width,
            )

            if isinstance(result, np.ndarray):
                result[gap_radii] = 0.0
            elif gap_radii:
                result = 0.0

            return result

        return wrapper_inner

    return wrapper_outer


def accretion_disc_self_similar(
    radius: np.ndarray, radius_critical: float, gamma: float
):
    """
    Lynden-Bell and Pringle (1974) self-similar solution.

    (R / R_crit) ^ (-y) * exp[ - (R / R_crit) ^ (2 - y)

    Parameters
    ----------
    radius
        The radius at which to evaulate the function.
    radius_critical
        The critical radius for the exponential taper.
    gamma
        The exponent in the power law.

    Returns
    -------
    float
        The surface density at the specified radius.
    """
    rc, y = radius_critical, gamma
    return (radius / rc) ** (-y) * np.exp(-(radius / rc) ** (2 - y))
