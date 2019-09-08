from typing import Tuple

import numpy as np

from .defaults import options

_AVAILABLE_DISTRIBUTIONS = ('cubic', 'close packed')

HFACT_DEFAULT = options['hfact']


def uniform_distribution(
    *,
    box_dimensions: Tuple[float],
    particle_spacing: float,
    distribution: str = None,
    hfact: float = None,
):
    """
    Generate a uniform particle distribution in a Cartesian box.

    Parameters
    ----------
    box_dimensions : tuple
        The box dimensions as a tuple:
        (xmin, xmax, ymin, ymax, zmin, zmax).
    particle_spacing : float
        The spacing between the particles.

    Optional parameters
    -------------------
    distribution : str
        The type of distribution. Options: 'cubic' or 'close packed'.
        Default is 'close packed'.
    hfact : float
        The smoothing length factor. Default is 1.2.

    Returns
    -------
    position : (N, 3) np.ndarray
        The particle Cartesian positions.
    smoothing_length :(N,) np.ndarray
        The particle smoothing length.
    """

    if distribution is not None and distribution not in _AVAILABLE_DISTRIBUTIONS:
        raise ValueError('distribution not available')

    if distribution is None:
        distribution = 'close packed'

    if hfact is None:
        hfact = HFACT_DEFAULT

    xmin, xmax, ymin, ymax, zmin, zmax = box_dimensions

    xwidth = xmax - xmin
    ywidth = ymax - ymin
    zwidth = zmax - zmin

    if distribution == 'cubic':

        nx = int(xwidth / particle_spacing)
        ny = int(ywidth / particle_spacing)
        nz = int(zwidth / particle_spacing)

        dx = xwidth / nx
        dy = ywidth / ny
        dz = zwidth / nz

        x = np.linspace(xmin + dx / 2, xmax - dx / 2, nx)
        y = np.linspace(ymin + dy / 2, ymax - dy / 2, ny)
        z = np.linspace(zmin + dz / 2, zmax - dz / 2, nz)

        xx, yy, zz = np.meshgrid(x, y, z)

        position = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T

    if distribution == 'close packed':

        dx = particle_spacing
        dy = dx * np.sqrt(3) / 2
        dz = dx * (2 / 3 * np.sqrt(6)) / 2

        nx = int(xwidth / dx)
        ny = int(ywidth / dy)
        nz = int(zwidth / dz)

        position = _close_packed_lattice(nx, ny, nz) * particle_spacing / 2

        position[:, 0] -= position[:, 0].mean()
        position[:, 1] -= position[:, 1].mean()
        position[:, 2] -= position[:, 2].mean()

        position[:, 0] += (xmin + xmax) / 2
        position[:, 1] += (ymin + ymax) / 2
        position[:, 2] += (zmin + zmax) / 2

    smoothing_length = hfact * particle_spacing * np.ones(nx * ny * nz)

    return position, smoothing_length


def _close_packed_lattice(nx, ny, nz):

    pos = np.zeros((nx * ny * nz, 3))

    for k in range(nz):

        k_start = k * nx * ny
        k_end = k * nx * ny + nx * ny
        pos[k_start:k_end, 2] = 2 * np.sqrt(6) / 3 * k

        for j in range(ny):

            j_start = k_start + j * nx
            j_end = k_start + j * nx + nx
            pos[j_start:j_end, 1] = np.sqrt(3) * (j + 1 / 3 * np.mod(k, 2))

            for i in range(nx):

                pos[k * nx * ny + j * nx + i, 0] = 2 * i + np.mod(j + k, 2)

    return pos
