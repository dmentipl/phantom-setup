from typing import Callable, Tuple

import numpy as np


class Disc:
    def __init__(self):
        self._positions = None
        self._velocities = None

    @property
    def positions(self) -> np.ndarray:
        """Particle positions."""
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """Particle velocities."""
        return self._velocities

    def add_particles(self):
        self.set_positions()
        self.set_velocities()

    def set_positions(
        self,
        number_of_particles: float,
        density_distribution: Callable[[float], float],
        radius_range: Tuple[float, float],
        q_index: float,
        aspect_ratio: float,
        reference_radius: float,
        centre_of_mass: Tuple[float, float, float] = None,
        args: tuple = None,
    ) -> None:
        """
        Set the disc particle positions.

        Parameters
        ----------
        number_of_particles
            The number of particles.
        density_distribution
            The surface density as a function of radius.
        radius_range
            The range of radii as (R_min, R_max).
        q_index
            The index in the sound speed power law such that
                H ~ (R / R_reference) ^ (3/2 - q).
        aspect_ratio
            The aspect ratio at the reference radius.
        reference_radius
            The radius at which the aspect ratio is given.
        centre_of_mass
            The centre of mass of the disc, i.e. around which position
            is it rotating.
        args
            Extra arguments to pass to density_distribution.
        """

        # TODO:
        # - change orientation
        # - add warps
        # - support for external forces

        if centre_of_mass is None:
            centre_of_mass = (0.0, 0.0, 0.0)

        r_min = radius_range[0]
        r_max = radius_range[1]
        size = number_of_particles

        xi = np.sort(np.random.uniform(r_min, r_max, size))
        p = density_distribution(xi, *args)
        p /= np.sum(p)

        r = np.random.choice(xi, size=size, p=p)
        phi = np.random.rand(size) * 2 * np.pi
        H = (
            reference_radius ** (q_index - 1 / 2)
            * aspect_ratio
            * r ** (3 / 2 - q_index)
        )
        z = np.random.normal(scale=H)

        self._positions = np.array([r * np.cos(phi), r * np.sin(phi), z]).T
        self._positions += centre_of_mass

    def set_velocities(self):
        raise NotImplementedError
