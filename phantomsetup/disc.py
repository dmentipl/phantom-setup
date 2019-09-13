from typing import Callable, Tuple

import numpy as np


class Disc:
    def __init__(self):
        self._particle_positions = None
        self._particle_velocities = None

    @property
    def particle_positions(self) -> np.ndarray:
        """Particle positions."""
        return self._particle_positions

    @property
    def particle_velocities(self) -> np.ndarray:
        """Particle velocities."""
        return self._particle_velocities

    def add_particles(self):
        self.set_positions()
        self.set_velocities()

    def set_positions(
        self,
        number_of_particles: float,
        density_distribution: Callable[float],
        radius_range: Tuple[float, float],
        sound_speed: Callable[float],
        omega: Callable[float],
        centre_of_mass: Tuple[float, float, float] = None,
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
        sound_speed
            The sound speed as a function of radius.
        omega
            The Keplerian angular velocity as a function of radius.
        centre_of_mass
            The centre of mass of the disc, i.e. around which position
            is it rotating.
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
        p = density_distribution(xi)
        p /= np.sum(p)

        r = np.random.choice(xi, size=size, p=p)
        phi = np.random.rand(size) * 2 * np.pi
        H = sound_speed(r) / omega(r)
        z = np.random.normal(scale=H)

        self._particle_positions = np.array([r * np.cos(phi), r * np.sin(phi), z]).T
        self._particle_positions += centre_of_mass

    def set_velocities(self):
        raise NotImplementedError
