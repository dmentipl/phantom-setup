from typing import Callable, Tuple, Union

import numpy as np
from scipy import spatial


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
        rotation_axis: Union[Tuple[float, float, float], np.ndarray] = None,
        rotation_angle: float = None,
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

        Optional Parameters
        -------------------
        centre_of_mass
            The centre of mass of the disc, i.e. around which position
            is it rotating.
        rotation_axis
            An axis around which to rotate the disc.
        rotation_angle
            The angle to rotate around the rotation_axis.
        args
            Extra arguments to pass to density_distribution.
        """

        # TODO:
        # - change orientation
        # - add warps
        # - support for external forces

        if (rotation_axis is not None) ^ (rotation_angle is not None):
            raise ValueError(
                'Must specify rotation_angle and rotation_axis to perform rotation'
            )

        if rotation_axis is not None:
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation = spatial.transform.Rotation.from_rotvec(
                rotation_angle * rotation_axis
            )

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

        xyz = np.array([r * np.cos(phi), r * np.sin(phi), z]).T

        if rotation_axis is not None:
            xyz = rotation.apply(xyz)

        xyz += centre_of_mass

        self._positions = xyz

    def set_velocities(self):
        raise NotImplementedError
