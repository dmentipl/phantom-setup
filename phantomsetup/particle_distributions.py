from typing import Tuple

import numpy as np

_AVAILABLE_DISTRIBUTIONS = ('cubic', 'close packed')


def uniform_distribution(
    distribution: str, box_dimensions: Tuple[float], particle_spacing: float
):
    """
    Generate a uniform particle distribution.

    Parameters
    ----------
    distribution : str
        The type of distribution. Options: 'cubic'.
    box_dimensions : tuple
        The box dimensions as a tuple like:
        (xmin, xmax, ymin, ymax, zmin, zmax).
    particle_spacing : float
        The spacing between the particles.

    Returns
    -------
    (N, 3) np.ndarray
        An array of Cartesian particle positions.
    """

    if distribution not in _AVAILABLE_DISTRIBUTIONS:
        raise ValueError('distribution not available')

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

        nx = int(xwidth / particle_spacing)
        ny = int(ywidth / particle_spacing)
        nz = int(zwidth / particle_spacing)

        return _close_packed_lattice(nx, ny, nz)


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
