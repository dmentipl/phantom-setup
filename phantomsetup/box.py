from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from . import defaults
from .particles import Particles

_AVAILABLE_DISTRIBUTIONS = ('cubic', 'close packed')
_HFACT_DEFAULT = defaults.run_options.config['hfact'].value


class Box(Particles):
    """
    Cartesian box for simulations.

    Parameters
    ----------
    xmin
        Minimum x value.
    xmax
        Maximum x value.
    ymin
        Minimum y value.
    ymax
        Maximum y value.
    zmin
        Minimum z value.
    zmax
        Maximum x value.
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
    ) -> None:
        super().__init__()

        self._boundary = (xmin, xmax, ymin, ymax, zmin, zmax)

        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._zmin = zmin
        self._zmax = zmax

        self._xwidth = xmax - xmin
        self._ywidth = ymax - ymin
        self._zwidth = zmax - zmin

        self._volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

        self._particle_type: float = None
        self._particle_mass: float = None

    @property
    def boundary(self) -> Tuple[float, float, float, float, float, float]:
        """Box boundary (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return self._boundary

    @property
    def volume(self) -> float:
        """Box volume."""
        return self._volume

    @property
    def xmin(self) -> float:
        return self._xmin

    @property
    def xmax(self) -> float:
        return self._xmax

    @property
    def ymin(self) -> float:
        return self._ymin

    @property
    def ymax(self) -> float:
        return self._ymax

    @property
    def zmin(self) -> float:
        return self._zmin

    @property
    def zmax(self) -> float:
        return self._zmax

    @property
    def xwidth(self) -> float:
        """Box width in x-direction."""
        return self._xwidth

    @property
    def ywidth(self) -> float:
        """Box width in y-direction."""
        return self._ywidth

    @property
    def zwidth(self) -> float:
        """Box width in z-direction."""
        return self._zwidth

    def add_particles(
        self,
        particle_type: int,
        number_of_particles: int,
        density: float,
        velocity_distribution: Callable[[np.ndarray], np.ndarray],
        hfact: float = None,
    ) -> Box:
        """
        Add uniform particle distribution with arbitrary velocity field.

        Parameters
        ----------
        setup
            The phantomsetup object representing the simulation.
        particle_type
            The particle type to add.
        number_of_particles
            The number of particles to add.
        density
            The initial uniform density.
        velocity_distribution
            The initial velocity distribution as a function taking an
            array of positions with shape (N, 3) and returning an array
            of velocities with shape (N, 3).

        Optional Parameters
        -------------------
        hfact
            The smoothing length factor.
        """

        if hfact is None:
            hfact = defaults.run_options.config['hfact'].value

        particle_spacing = (self.volume / number_of_particles) ** (1 / 3)

        position, smoothing_length = uniform_distribution(
            boundary=self.boundary, particle_spacing=particle_spacing, hfact=hfact
        )

        particle_mass = density * self.volume / number_of_particles

        velocity = velocity_distribution(position)

        self._particle_type = particle_type
        self._particle_mass = particle_mass
        self._position = position
        self._velocity = velocity
        self._smoothing_length = smoothing_length

        return


def uniform_distribution(
    *,
    boundary: Tuple[float, float, float, float, float, float],
    particle_spacing: float,
    distribution: str = None,
    hfact: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a uniform particle distribution in a Cartesian box.

    Parameters
    ----------
    boundary
        The boundary as a tuple (xmin, xmax, ymin, ymax, zmin, zmax).
    particle_spacing
        The spacing between the particles.

    Optional parameters
    -------------------
    distribution
        The type of distribution. Options: 'cubic' or 'close packed'.
        Default is 'close packed'.
    hfact
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
        hfact = _HFACT_DEFAULT

    xmin, xmax, ymin, ymax, zmin, zmax = boundary

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


def _close_packed_lattice(nx: int, ny: int, nz: int) -> np.ndarray:

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
