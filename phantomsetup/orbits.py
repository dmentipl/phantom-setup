from typing import NewType, Tuple

import numpy as np


def hill_sphere_radius(
    planet_radius: float,
    planet_mass: float,
    stellar_mass: float,
    eccentricity: float = None,
) -> float:
    """
    Calculate the Hill sphere radius.

    Parameters
    ----------
    planet_radius
        The orbital radius of the planet.
    planet_mass
        The mass of the planet.
    stellar_mass
        The mass of the star.

    Optional Parameters
    -------------------
    eccentricity
        The orbital eccentricity.

    Returns
    -------
    hill_radius
        The Hill sphere radius.
    """

    if eccentricity is None:
        eccentricity = 0.0

    return (
        (1 - eccentricity)
        * planet_radius
        * (planet_mass / (3 * stellar_mass)) ** (1 / 3)
    )


ThreeTuple = NewType('ThreeTuple', int)


def binary_position_velocity(
    primary_mass: float,
    secondary_mass: float,
    semi_major_axis: float,
    eccentricity: float,
    inclination: float = None,
    longitude_ascending_node: float = None,
    argument_periapsis: float = None,
    true_anomaly: float = None,
    use_degrees: bool = True,
) -> Tuple[ThreeTuple, ThreeTuple, ThreeTuple, ThreeTuple]:
    """
    Set position and velocity of two bodies in a bound Keplerian orbit.

    Parameters
    ----------
    primary_mass (m1)
        The mass of the primary.
    secondary_mass (m2)
        The mass of the secondary.
    semi_major_axis (a)
        The semi major axis.
    eccentricity (e)
        The orbital eccentricity.

    Optional Parameters
    -------------------
    inclination (i)
        The orbital inclination. Default is 0.0.
    longitude_ascending_node (Omega)
        The longitude of ascending node. Default is 0.0.
    argument_periapsis (pomega)
        The argument of periapsis. Default is 0.0.
    true_anomaly (f)
        The true anomaly. Default is 0.0.
    use_degrees
        If true, specify angles in degrees. Otherwise, use radians.

    Returns
    -------
    primary_position
        The Cartesian position of the primary.
    secondary_position
        The Cartesian position of the secondary.
    primary_velocity
        The Cartesian velocity of the primary.
    secondary_velocity
        The Cartesian velocity of the secondary.
    """

    m1 = primary_mass
    m2 = secondary_mass
    a = semi_major_axis
    e = eccentricity

    if inclination is None:
        i = 0.0
    else:
        i = inclination

    if longitude_ascending_node is None:
        Omega = 0.0
    else:
        Omega = longitude_ascending_node

    if argument_periapsis is None:
        pomega = 0.0
    else:
        pomega = argument_periapsis

    if true_anomaly is None:
        f = 0.0
    else:
        f = true_anomaly

    if use_degrees:
        i *= np.pi / 180
        Omega *= np.pi / 180
        pomega *= np.pi / 180
        f *= np.pi / 180

    # Phantom convention: longitude of the ascending node is measured east of north
    Omega += np.pi / 2

    dx = 0.0
    dv = 0.0

    E = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(f), (e + np.cos(f)))

    P = np.zeros(3)
    Q = np.zeros(3)

    P[0] = np.cos(pomega) * np.cos(Omega) - np.sin(pomega) * np.cos(i) * np.sin(Omega)
    P[1] = np.cos(pomega) * np.sin(Omega) + np.sin(pomega) * np.cos(i) * np.cos(Omega)
    P[2] = np.sin(pomega) * np.sin(i)
    Q[0] = -np.sin(pomega) * np.cos(Omega) - np.cos(pomega) * np.cos(i) * np.sin(Omega)
    Q[1] = -np.sin(pomega) * np.sin(Omega) + np.cos(pomega) * np.cos(i) * np.cos(Omega)
    Q[2] = np.sin(i) * np.cos(pomega)

    E_dot = np.sqrt((m1 + m2) / (a ** 3)) / (1 - e * np.cos(E))

    dx = a * ((np.cos(E) - e) * P + np.sqrt(1.0 - (e ** 2)) * np.sin(E) * Q)

    dv = (
        -a * np.sin(E) * E_dot * P + a * np.sqrt(1.0 - (e ** 2)) * np.cos(E) * E_dot * Q
    )

    # Set centre of mass to zero
    primary_position = -dx * m2 / (m1 + m2)
    secondary_position = dx * m1 / (m1 + m2)
    primary_velocity = -dv * m2 / (m1 + m2)
    secondary_velocity = dv * m1 / (m1 + m2)

    return primary_position, secondary_position, primary_velocity, secondary_velocity
