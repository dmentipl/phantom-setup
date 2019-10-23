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
    """
    Stretch mapping.

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

    if geometry is 'None':
        geometry = 'cartesian'
    geometry = geometry.lower()
    if geometry not in _GEOMETRIES:
        raise ValueError(
            '"geometry" must be in ("cartesian", "cylindrical", "spherical")'
        )
    if coordinate is 'None':
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
        _positions = coordinate_transform(
            positions, geometry_from='cartesian', geometry_to=geometry
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
            _positions, geometry_from=geometry, geometry_to='cartesian'
        )
    else:
        positions = _positions
    return positions


def coordinate_transform(
    position: ndarray, geometry_from: str, geometry_to: str, in_place: bool = False
) -> Optional[ndarray]:
    """
    Coordinate transformation.

    Transform 3d coordinates from one system to another. Coordinate
    systems supported: 'cartesian', 'cylindrical', 'spherical'.

    Parameters
    ----------
    coordinates
        The 3d coordinates to transform, as a (N, 3) ndarray.
    geometry_from
        The geometry that the coordinates are in.
    geometry_to
        The geometry to convert to.
    in_place, optional
        If True, the coordinate transformation operates on the the
        array in place, and the function returns None. Default: False.

    Returns
    -------
    Optional[ndarray]
        The coordinates transformed to the new coordinate system, if
        in_place is False. Otherwise, return None.
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
            return _cartesian_to_cylindrical(position, in_place)
        elif geometry_to == 'spherical':
            return _cartesian_to_spherical(position, in_place)
    if geometry_from == 'spherical':
        if geometry_to == 'cartesian':
            return _spherical_to_cartesian(position, in_place)
        else:
            raise ValueError('Can only convert spherical to cartesian')
    if geometry_from == 'cylindrical':
        if geometry_to == 'cartesian':
            return _cylindrical_to_cartesian(position, in_place)
        else:
            raise ValueError('Can only convert cylindrical to cartesian')
    raise ValueError('Failed to perform coordinate transform')


def _cartesian_to_cylindrical(
    position: ndarray, in_place: bool = False
) -> Optional[ndarray]:
    if in_place:
        r = np.hypot(position[:, 0], position[:, 1])
        phi = np.arctan2(position[:, 1], position[:, 0])
        position[:, 0] = r
        position[:, 1] = phi
        return None
    _position = np.zeros(position.shape)
    _position[:, 0] = np.hypot(position[:, 0], position[:, 1])
    _position[:, 1] = np.arctan2(position[:, 1], position[:, 0])
    _position[:, 2] = position[:, 2]
    return _position


def _cylindrical_to_cartesian(
    position: ndarray, in_place: bool = False
) -> Optional[ndarray]:
    if in_place:
        x = position[:, 0] * np.cos(position[:, 1])
        y = position[:, 0] * np.sin(position[:, 1])
        position[:, 0] = x
        position[:, 1] = y
        return None
    _position = np.zeros(position.shape)
    _position[:, 0] = position[:, 0] * np.cos(position[:, 1])
    _position[:, 1] = position[:, 0] * np.sin(position[:, 1])
    _position[:, 2] = position[:, 2]
    return _position


def _cartesian_to_spherical(
    position: ndarray, in_place: bool = False
) -> Optional[ndarray]:
    if in_place:
        x = position[:, 0]
        y = position[:, 1]
        z = position[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        position[:, 0] = r
        position[:, 1] = theta
        position[:, 2] = phi
        return None
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    _position = np.zeros(position.shape)
    _position[:, 0] = r
    _position[:, 1] = np.arccos(z / r)
    _position[:, 2] = np.arctan2(y, x)
    return _position
