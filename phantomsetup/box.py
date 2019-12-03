"""Box of particles."""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy import ndarray

from .boundary import Boundary
from .particles import Particles

_LATTICES = ('cubic', 'close packed')


class Box(Particles):
    """A uniformly distributed box of particles.

    Add uniform particle distribution with arbitrary velocity field.

    Parameters
    ----------
    box_boundary
        Boundary box as (xmin, xmax, ymin, ymax, zmin, zmax).
    particle_type
        The particle type to add.
    number_of_particles_in_x
        The number of particles in the x-direction to add.
    density
        The initial uniform density.
    velocity_distribution
        The initial velocity distribution as a function taking an
        array of position with shape (N, 3) and returning an array
        of velocity with shape (N, 3).

    Optional Parameters
    -------------------
    kwargs
        Keyword arguments to pass to uniform_distribution.
    """

    def __init__(
        self,
        box_boundary: Tuple[float, float, float, float, float, float],
        particle_type: int,
        number_of_particles_in_x: int,
        density: float,
        velocity_distribution: Callable[[ndarray, ndarray, ndarray], Tuple],
        **kwargs,
    ) -> None:
        super().__init__()

        boundary = Boundary(*box_boundary)
        particle_spacing = boundary.xwidth / number_of_particles_in_x

        position, smoothing_length = uniform_distribution(
            boundary=boundary.boundary, particle_spacing=particle_spacing, **kwargs
        )

        number_of_particles = len(smoothing_length)
        particle_mass = density * boundary.volume / number_of_particles

        velocity = np.zeros(position.shape)
        velocity[:, 0], velocity[:, 1], velocity[:, 2] = velocity_distribution(
            position[:, 0], position[:, 1], position[:, 2]
        )

        self.add_particles(
            particle_type=particle_type,
            particle_mass=particle_mass,
            position=position,
            velocity=velocity,
            smoothing_length=smoothing_length,
        )


def uniform_distribution(
    *,
    boundary: Tuple[float, float, float, float, float, float],
    particle_spacing: float,
    hfact: float = 1.2,
    lattice: str = None,
) -> Tuple[ndarray, ndarray]:
    """Generate a uniform particle distribution in a Cartesian box.

    Parameters
    ----------
    boundary
        The boundary as a tuple (xmin, xmax, ymin, ymax, zmin, zmax).
    particle_spacing
        The spacing between the particles.

    Optional parameters
    -------------------
    hfact
        The smoothing length factor. Default is 1.2.
    lattice
        The type of lattice. Options: 'cubic' or 'close packed'. Default
        is 'close packed'.

    Returns
    -------
    position : (N, 3) ndarray
        The particle Cartesian position.
    smoothing_length :(N,) ndarray
        The particle smoothing length.
    """
    if lattice is not None and lattice not in _LATTICES:
        raise ValueError('lattice not available')

    if lattice is None:
        lattice = 'close packed'

    xmin, xmax, ymin, ymax, zmin, zmax = boundary

    xwidth = xmax - xmin
    ywidth = ymax - ymin
    zwidth = zmax - zmin

    if lattice == 'cubic':

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

    elif lattice == 'close packed':

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

    else:
        raise ValueError('Cannot determine lattice')

    smoothing_length = hfact * particle_spacing * np.ones(nx * ny * nz)

    return position, smoothing_length


def _close_packed_lattice(nx: int, ny: int, nz: int) -> ndarray:

    xyz = np.zeros((nx * ny * nz, 3))

    for k in range(nz):

        k_start = k * nx * ny
        k_end = k * nx * ny + nx * ny
        xyz[k_start:k_end, 2] = 2 * np.sqrt(6) / 3 * k

        for j in range(ny):

            j_start = k_start + j * nx
            j_end = k_start + j * nx + nx
            xyz[j_start:j_end, 1] = np.sqrt(3) * (j + 1 / 3 * np.mod(k, 2))

            for i in range(nx):

                xyz[k * nx * ny + j * nx + i, 0] = 2 * i + np.mod(j + k, 2)

    return xyz
