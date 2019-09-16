from typing import Tuple

import numpy as np
from scipy import spatial


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


def binary_orbit(
    primary_mass: float,
    secondary_mass: float,
    semi_major_axis: float,
    eccentricity: float,
    inclination: float = None,
    longitude_ascending_node: float = None,
    argument_periapsis: float = None,
    true_anomaly: float = None,
    use_degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Set position and velocity of two bodies in an elliptic orbit.

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


def flyby_orbit(
    primary_mass: float,
    secondary_mass: float,
    periapsis_distance: float,
    initial_distance_in_peri_units: float = None,
    inclination: float = None,
    longitude_ascending_node: float = None,
    use_degrees: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Set position and velocity of two bodies in a parabolic orbit.

    The primary is set at the origin with zero initial velocity, and the
    secondary, or "perturber", is at some distance with non-zero initial
    velocity such that the binary orbit is parabolic.

    Parameters
    ----------
    primary_mass (m1)
        The mass of the primary.
    secondary_mass (m2)
        The mass of the secondary.
    periapsis_distance (rp)
        The distance of closest approach. Half the semi-latus rectum.

    Optional Parameters
    -------------------
    initial_distance_in_peri_units (n0)
        The initial distance in units or periapsis. Default is 10.0.
    inclination (i)
        The orbital inclination. Default is 0.0.
    longitude_ascending_node (Omega)
        The longitude of ascending node. Default is 0.0.
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
    rp = periapsis_distance

    if initial_distance_in_peri_units is None:
        n0 = 10.0
    else:
        n0 = initial_distance_in_peri_units

    if longitude_ascending_node is None:
        Omega = 0.0
    else:
        Omega = longitude_ascending_node

    if inclination is None:
        i = 0.0
    else:
        i = inclination

    if use_degrees:
        i *= np.pi / 180
        Omega *= np.pi / 180

    # Primary position and velocity
    primary_position = np.array((0.0, 0.0, 0.0))
    primary_velocity = np.array((0.0, 0.0, 0.0))

    # Focal parameter: pf
    pf = 2 * rp

    # Define m0 = -x0 / rp such that r0 = n0 * rp
    # the secondary starts at negative x and y
    # positive root of 1/8*m**4 + m**2 + 2(1-n0**2) = 0
    # for n0 > 1
    m0 = 2 * np.sqrt(n0 - 1)

    # Secondary initial position
    x = -m0 * rp
    y = rp * (1 - (x / pf) ** 2)
    z = 0.0
    secondary_position = np.array((x, y, z))

    # Secondary initial velocity
    r = np.linalg.norm(secondary_position)
    vx = (1.0 + (y / r)) * np.sqrt((m1 + m2) / pf)
    vy = -(x / r) * np.sqrt((m1 + m2) / pf)
    vz = 0.0
    secondary_velocity = np.array((vx, vy, vz))

    # Incline orbit about ascending node
    # i=0: prograde orbit
    # i=180: retrograde orbit
    # Convention: clock-wise rotation in the zx-plane
    i = np.pi - i
    rotation_axis = np.array((np.sin(Omega), -np.cos(Omega), 0.0))
    rotation = spatial.transform.Rotation.from_rotvec(i * rotation_axis)

    secondary_position = rotation.apply(secondary_position)
    secondary_velocity = rotation.apply(secondary_velocity)

    return primary_position, secondary_position, primary_velocity, secondary_velocity


def flyby_time(
    primary_mass: float,
    secondary_mass: float,
    periapsis_distance: float,
    gravitational_constant: float,
    initial_distance_in_peri_units: float = None,
) -> float:
    """
    Calculate flyby time.

    Determine the time from initial separation, to the equivalent
    separation past periastron using Barker's equation.

    Parameters
    ----------
    primary_mass (m1)
        The mass of the primary.
    secondary_mass (m2)
        The mass of the secondary.
    periapsis_distance (rp)
        The distance of closest approach. Half the semi-latus rectum.
    gravitational_constant (G)
        The gravitational constant.

    Optional Parameters
    -------------------
    initial_distance_in_peri_units (n0)
        The initial distance in units or periapsis. Default is 10.0.

    Returns
    -------
    The flyby time.
    """

    m1 = primary_mass
    m2 = secondary_mass
    rp = periapsis_distance
    G = gravitational_constant

    if initial_distance_in_peri_units is None:
        n0 = 10.0
    else:
        n0 = initial_distance_in_peri_units

    # Semi-latus rectum
    p = 2 * rp

    # Initial position
    xi = -2 * np.sqrt(n0 - 1.0) * rp
    yi = rp * (1 - (xi / p) ** 2)

    # Graviational parameter
    mu = G * (m1 + m2)

    # True anomaly
    nu = np.pi - np.arctan(np.abs(xi / yi))

    # Barker's equation
    Di = np.tan(-nu / 2)
    Df = np.tan(nu / 2)
    T = 1 / 2 * np.sqrt(p ** 3 / mu) * (Df + 1 / 3 * Df ** 3 - Di - 1 / 3 * Di ** 3)

    return T
