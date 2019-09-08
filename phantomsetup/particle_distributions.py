from typing import Tuple

import numpy as np

_AVAILABLE_DISTRIBUTIONS = ('cubic', 'close packed')


def uniform_distribution(
    *, box_dimensions: Tuple[float], particle_spacing: float, distribution: str = None
):
    """
    Generate a uniform particle distribution.

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

    Returns
    -------
    (N, 3) np.ndarray
        An array of Cartesian particle positions.
    """

    if distribution is not None and distribution not in _AVAILABLE_DISTRIBUTIONS:
        raise ValueError('distribution not available')

    if distribution is None:
        distribution = 'close packed'

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

        return np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T

    if distribution == 'close packed':

        dx = particle_spacing
        dy = dx * np.sqrt(3) / 2
        dz = dx * (2 / 3 * np.sqrt(6)) / 2

        nx = int(xwidth / dx)
        ny = int(ywidth / dy)
        nz = int(zwidth / dz)

        pos = _close_packed_lattice(nx, ny, nz) * particle_spacing / 2

        pos[:, 0] -= pos[:, 0].mean()
        pos[:, 1] -= pos[:, 1].mean()
        pos[:, 2] -= pos[:, 2].mean()

        pos[:, 0] += (xmin + xmax) / 2
        pos[:, 1] += (ymin + ymax) / 2
        pos[:, 2] += (zmin + zmax) / 2

        return pos


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
