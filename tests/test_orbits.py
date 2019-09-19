import numpy as np
import pytest

from phantomsetup import orbits


@pytest.mark.parametrize(
    'planet_radius, planet_mass, stellar_mass, eccentricity, r_hill_exact',
    [(1.0, 0.01, 1.0, 0.0, 0.149380158)],
)
def test_hill_sphere_radius(
    planet_radius, planet_mass, stellar_mass, eccentricity, r_hill_exact
):
    r_hill = orbits.hill_sphere_radius(
        planet_radius=planet_radius,
        planet_mass=planet_mass,
        stellar_mass=stellar_mass,
        eccentricity=eccentricity,
    )

    result = np.allclose(r_hill, r_hill_exact)
    assert result


@pytest.mark.parametrize(
    (
        'primary_mass, secondary_mass, semi_major_axis, eccentricity, '
        'pos1_exact, pos2_exact, vel1_exact, vel2_exact'
    ),
    [
        (
            1.0,
            1.0,
            1.0,
            0.0,
            [0, -0.5, 0],
            [0, 0.5, 0],
            [0.707106781, 0, 0],
            [-0.707106781, 0, 0],
        ),
        (
            1.0,
            1.0,
            1.0,
            0.5,
            [0, -0.25, 0],
            [0, 0.25, 0],
            [1.22474487, 0, 0],
            [-1.22474487, 0, 0],
        ),
    ],
)
def test_binary_orbit(
    primary_mass,
    secondary_mass,
    semi_major_axis,
    eccentricity,
    pos1_exact,
    pos2_exact,
    vel1_exact,
    vel2_exact,
):

    pos1, pos2, vel1, vel2 = orbits.binary_orbit(
        primary_mass, secondary_mass, semi_major_axis, eccentricity
    )

    result = np.allclose(pos1, pos1_exact)
    assert result
    result = np.allclose(pos2, pos2_exact)
    assert result
    result = np.allclose(vel1, vel1_exact)
    assert result
    result = np.allclose(vel2, vel2_exact)
    assert result


@pytest.mark.parametrize(
    'primary_mass, secondary_mass, periapsis_distance, pos1_exact, pos2_exact, '
    'vel1_exact, vel2_exact',
    [(1, 1, 10, [0, 0, 0], [60, -80, 0], [0, 0, 0], [-6.324555e-2, 1.8973666e-1, 0])],
)
def test_flyby_orbit(
    primary_mass,
    secondary_mass,
    periapsis_distance,
    pos1_exact,
    pos2_exact,
    vel1_exact,
    vel2_exact,
):

    pos1, pos2, vel1, vel2 = orbits.flyby_orbit(
        primary_mass, secondary_mass, periapsis_distance
    )

    result = np.allclose(pos1, pos1_exact)
    assert result
    result = np.allclose(pos2, pos2_exact)
    assert result
    result = np.allclose(vel1, vel1_exact)
    assert result
    result = np.allclose(vel2, vel2_exact)
    assert result


@pytest.mark.parametrize(
    'primary_mass, secondary_mass, periapsis_distance, gravitational_constant, '
    'time_exact',
    [(1.0, 1.0, 10.0, 1.0, 758.946638440411)],
)
def test_flyby_time(
    primary_mass, secondary_mass, periapsis_distance, gravitational_constant, time_exact
):
    time = orbits.flyby_time(
        primary_mass, secondary_mass, periapsis_distance, gravitational_constant
    )

    result = np.allclose(time, time_exact)
    assert result
