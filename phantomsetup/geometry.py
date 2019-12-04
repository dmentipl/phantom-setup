"""Geometry utilities.

For example, stretch mapping and coordinate transformations.
"""

from typing import Callable, Optional

import numba
import numpy as np
from numba import float64
from numpy import ndarray
from scipy import optimize

_GEOMETRIES = ('cartesian', 'cylindrical', 'spherical')


def stretch_map(
    density_function: Callable[[float], float],
    positions: ndarray,
    coordinate_min: float,
    coordinate_max: float,
    geometry: str = 'None',
    coordinate: str = 'None',
) -> ndarray:
    """Stretch mapping.

    Deform a uniform particle distribution in one dimension with an
    arbitrary scalar function.

    Parameters
    ----------
    density_function
        The scalar function. This should be a function of one variable.
        It is best to vectorize it via numba.vectorize, or it will be
        slow. See notes below.
    positions
        The uniform particle positions in Cartesian form as a (N, 3)
        ndarray.
    coordinate_min
        The minimum coordinate value for the stretch mapping.
    coordinate_max
        The maximum coordinate value for the stretch mapping.
    geometry
        The geometry: either 'cartesian', 'cylindrical', or 'spherical'.
    coordinate
        The coordinate for the function. Options are: 'x', 'y', 'z',
        'r', 'phi', 'theta'.

    Returns
    -------
    ndarray
        The particle positions after the stretch mapping.

    Notes
    -----
    To make a fast numba function, write a function as if for a scalar
    value then decorate it with @numba.vectorize([float64(float64)]).
    For example
    >>> @numba.vectorize([float64(float64)])
    ... def my_func(x):
    ...     return 1 + np.sin(x) ** 2
    """
    if geometry == 'None':
        geometry = 'cartesian'
    geometry = geometry.lower()
    if geometry not in _GEOMETRIES:
        raise ValueError(
            '"geometry" must be in ("cartesian", "cylindrical", "spherical")'
        )
    if coordinate == 'None':
        if geometry == 'cartesian':
            coordinate = 'x'
        elif geometry in ('cylindrical', 'spherical'):
            coordinate = 'r'
    coordinate = coordinate.lower()
    if coordinate not in ('x', 'y', 'z', 'r', 'phi', 'theta'):
        raise ValueError('"coordinate" must be in ("x", "y", "z", "r", "phi", "theta")')
    if geometry == 'spherical' and coordinate == 'r':
        if coordinate_min < 0.0:
            raise ValueError('"coordinate_min" < 0.0 for radius: not physical')
        i_area_element = 3
    elif geometry == 'cylindrical' and coordinate == 'r':
        if coordinate_min < 0.0:
            raise ValueError('"coordinate_min" < 0.0 for radius: not physical')
        i_area_element = 2
    else:
        i_area_element = 1

    if i_area_element == 1:

        @numba.vectorize([float64(float64)])
        def rho_dS(x):
            rho_dS = density_function(x)
            return rho_dS

    elif i_area_element == 2:

        @numba.vectorize([float64(float64)])
        def rho_dS(x):
            rho_dS = 2 * np.pi * x * density_function(x)
            return rho_dS

    elif i_area_element == 3:

        @numba.vectorize([float64(float64)])
        def rho_dS(x):
            rho_dS = 4 * np.pi * x ** 2 * density_function(x)
            return rho_dS

    @numba.vectorize([float64(float64, float64)])
    def mass(x, x_min):
        _x = np.linspace(x_min, x)
        m = np.trapz(rho_dS(_x), _x)
        return m

    def func(x, x_min, x_max, x_original):
        f = mass(x, x_min) / mass(x_max, x_min) - (x_original - x_min) / (x_max - x_min)
        return f

    def dfunc(x, x_min, x_max, x_original):
        df = rho_dS(x) / mass(x_max, x_min)
        return df

    if geometry in ('cylindrical', 'spherical'):
        _positions: ndarray = coordinate_transform(
            position=positions, geometry_from='cartesian', geometry_to=geometry
        )
    else:
        _positions = np.copy(positions)

    x_original = _positions[:, 0]
    x_min = coordinate_min
    x_max = coordinate_max
    x_guess = x_original
    x_stretched = optimize.newton(
        func, x_guess, fprime=dfunc, args=(x_min, x_max, x_original)
    )

    _positions[:, 0] = x_stretched
    if geometry in ('cylindrical', 'spherical'):
        positions = coordinate_transform(
            position=_positions, geometry_from=geometry, geometry_to='cartesian'
        )
    else:
        positions = _positions
    return positions


def coordinate_transform(
    *,
    position: ndarray,
    velocity: ndarray = None,
    geometry_from: str,
    geometry_to: str,
    in_place: bool = False,
) -> Optional[ndarray]:
    """
    Coordinate transformation.

    Transform 3d coordinates from one system to another. Coordinate
    systems supported: 'cartesian', 'cylindrical', 'spherical'.
    Performs transformation of positions and, optionally, velocities.

    Parameters
    ----------
    position
        The 3d coordinates to transform, as a (N, 3) ndarray.
    velocity, optional
        The 3d velocity components to transform, as a (N, 3) ndarray.
    geometry_from
        The geometry that the coordinates are in.
    geometry_to
        The geometry to convert to.
    in_place, optional
        If True, the coordinate transformation operates on the the
        array in place, and the function returns None. Default: False.

    Returns
    -------
    position : ndarray
        The coordinates transformed to the new coordinate system, if
        in_place is False. Otherwise, return None.
    velocity : ndarray
        The velocity components transformed to the new coordinate
        system, if in_place is False, and if velocity is not None.
        Otherwise, return None.
    """
    _GEOMETRIES = ('cartesian', 'cylindrical', 'spherical')

    geometry_from = geometry_from.lower()
    geometry_to = geometry_to.lower()
    if geometry_from not in _GEOMETRIES:
        raise ValueError('"geometry_from" not available')
    if geometry_to not in _GEOMETRIES:
        raise ValueError('"geometry_to" not available')

    if geometry_from == 'cartesian':
        if geometry_to == 'cylindrical':
            return _cartesian_to_cylindrical(position, velocity, in_place)
        elif geometry_to == 'spherical':
            return _cartesian_to_spherical(position, velocity, in_place)
    if geometry_from == 'spherical':
        if geometry_to == 'cartesian':
            return _spherical_to_cartesian(position, velocity, in_place)
        else:
            raise ValueError('Can only convert spherical to cartesian')
    if geometry_from == 'cylindrical':
        if geometry_to == 'cartesian':
            return _cylindrical_to_cartesian(position, velocity, in_place)
        else:
            raise ValueError('Can only convert cylindrical to cartesian')
    raise ValueError('Failed to perform coordinate transform')


def _cartesian_to_cylindrical(
    position: ndarray, velocity: ndarray = None, in_place: bool = False
) -> Optional[ndarray]:
    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    if velocity is not None:
        vx, vy, vz = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        vr = (x * vx + y * vy) / r
        vphi = (x * vy - y * vx) / r ** 2
    if in_place:
        position[:, 0] = r
        position[:, 1] = phi
        if velocity is not None:
            velocity[:, 0] = vr
            velocity[:, 1] = vphi
        return None
    _position = np.zeros(position.shape)
    _position[:, 0] = r
    _position[:, 1] = phi
    _position[:, 2] = z
    if velocity is not None:
        _velocity = np.zeros(velocity.shape)
        _velocity[:, 0] = vr
        _velocity[:, 1] = vphi
        _velocity[:, 2] = vz
        return _position, _velocity
    return _position, None


def _cylindrical_to_cartesian(
    position: ndarray, velocity: ndarray = None, in_place: bool = False
) -> Optional[ndarray]:
    r, phi, z = position[:, 0], position[:, 1], position[:, 2]
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    if velocity is not None:
        vr, vphi, vz = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        vx = vr * np.cos(phi) - r * vphi * np.sin(phi)
        vy = vr * np.sin(phi) + r * vphi * np.cos(phi)
    if in_place:
        position[:, 0] = x
        position[:, 1] = y
        if velocity is not None:
            velocity[:, 0] = vx
            velocity[:, 1] = vy
        return None
    _position = np.zeros(position.shape)
    _position[:, 0] = x
    _position[:, 1] = y
    _position[:, 2] = z
    if velocity is not None:
        _velocity = np.zeros(velocity.shape)
        _velocity[:, 0] = vx
        _velocity[:, 1] = vy
        _velocity[:, 2] = vz
        return _position, _velocity
    return _position, None


def _cartesian_to_spherical(
    position: ndarray, velocity: ndarray = None, in_place: bool = False
) -> Optional[ndarray]:
    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    xy = np.hypot(x, y)
    r = np.hypot(xy, z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    if velocity is not None:
        vx, vy, vz = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        vr = (x * vx + y * vy + z * vz) / r
        vtheta = (r * vz - vr * z) / r ** 2
        vphi = (x * vy - vx * y) / xy ** 2
    if in_place:
        position[:, 0] = r
        position[:, 1] = theta
        position[:, 2] = phi
        if velocity is not None:
            velocity[:, 0] = vr
            velocity[:, 1] = vtheta
            velocity[:, 2] = vphi
        return None
    _position = np.zeros(position.shape)
    _position[:, 0] = r
    _position[:, 1] = theta
    _position[:, 2] = phi
    if velocity is not None:
        _velocity = np.zeros(position.shape)
        _velocity[:, 0] = vr
        _velocity[:, 1] = vtheta
        _velocity[:, 2] = vphi
        return _position, _velocity
    return _position, None


def _spherical_to_cartesian(
    position: ndarray, velocity: ndarray = None, in_place: bool = False
) -> Optional[ndarray]:
    r, theta, phi = position[:, 0], position[:, 1], position[:, 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    if velocity is not None:
        vr, vtheta, vphi = velocity[:, 0], velocity[:, 1], velocity[:, 2]
        vx = (
            vr * np.sin(theta) * np.cos(phi)
            + r * vtheta * np.cos(theta) * np.cos(phi)
            - r * vphi * np.sin(theta) * np.sin(phi)
        )
        vy = (
            vr * np.sin(theta) * np.sin(phi)
            + r * vtheta * np.cos(theta) * np.sin(phi)
            + r * vphi * np.sin(theta) * np.cos(phi)
        )
        vz = vr * np.cos(theta) - r * vtheta * np.sin(theta)
    if in_place:
        position[:, 0] = x
        position[:, 1] = y
        position[:, 2] = z
        if velocity is not None:
            velocity[:, 0] = vx
            velocity[:, 1] = vy
            velocity[:, 2] = vz
        return None
    _position = np.zeros(position.shape)
    _position[:, 0] = x
    _position[:, 1] = y
    _position[:, 2] = z
    if velocity is not None:
        _velocity = np.zeros(velocity.shape)
        _velocity[:, 0] = vx
        _velocity[:, 1] = vy
        _velocity[:, 2] = vz
        return _position, _velocity
    return _position, None
