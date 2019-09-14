from __future__ import annotations

from typing import Callable, Tuple, Union

import numpy as np
from scipy import integrate, spatial


class Disc:
    """
    Accretion disc.
    """

    def __init__(self):

        self._particle_type: int = None
        self._particle_mass: float = None
        self._number_of_particles: int = None

        self._disc_mass: float = None
        self._positions: np.ndarray = None
        self._velocities: np.ndarray = None

    @property
    def particle_type(self) -> int:
        """Particle type."""
        return self._particle_type

    @property
    def particle_mass(self) -> int:
        """Particle mass."""
        return self._particle_mass

    @property
    def number_of_particles(self) -> int:
        """Number of particles in disc."""
        return self._number_of_particles

    @property
    def disc_mass(self) -> float:
        """Total disc mass."""
        return self._disc_mass

    @property
    def positions(self) -> np.ndarray:
        """Particle positions."""
        return self._positions

    @property
    def velocities(self) -> np.ndarray:
        """Particle velocities."""
        return self._velocities

    def add_particles(
        self,
        particle_type: int,
        number_of_particles: float,
        disc_mass: float,
        density_distribution: Callable[[float], float],
        radius_range: Tuple[float, float],
        q_index: float,
        aspect_ratio: float,
        reference_radius: float,
        stellar_mass: float,
        gravitational_constant: float,
        centre_of_mass: Tuple[float, float, float] = None,
        rotation_axis: Union[Tuple[float, float, float], np.ndarray] = None,
        rotation_angle: float = None,
        args: tuple = None,
        pressureless: bool = False,
    ) -> Disc:
        """
        Add particles to disc.

        This method sets the particle type, mass, positions, and
        velocities.

        Parameters
        ----------
        particle_type
            The integer particle type.
        number_of_particles
            The number of particles.
        disc_mass
            The total disc mass.
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
        stellar_mass
            The mass of the central object the disc is orbiting.
        gravitational_constant
            The gravitational constant.

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
        pressureless
            Set to True if the particles are pressureless, i.e. dust.
        """

        self.set_positions(
            number_of_particles,
            density_distribution,
            radius_range,
            q_index,
            aspect_ratio,
            reference_radius,
            centre_of_mass,
            rotation_axis,
            rotation_angle,
            args,
        )

        self.set_velocities(
            stellar_mass,
            gravitational_constant,
            q_index,
            aspect_ratio,
            reference_radius,
            rotation_axis,
            rotation_angle,
            pressureless,
        )

        self._particle_type = particle_type

        integrated_mass = integrate.quad(
            lambda x: 2 * np.pi * x * density_distribution(x, *args),
            radius_range[0],
            radius_range[1],
        )[0]

        self._particle_mass = (disc_mass / integrated_mass) / number_of_particles
        self._disc_mass = disc_mass

        return self

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
    ) -> Disc:
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
        # - set particle mass from disc mass, or toomre q, or something else
        # - add warps
        # - support for external forces
        # - add correction for self-gravity

        if (rotation_axis is not None) ^ (rotation_angle is not None):
            raise ValueError(
                'Must specify rotation_angle and rotation_axis to perform rotation'
            )

        self._number_of_particles = number_of_particles

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
        if args is not None:
            p = density_distribution(xi, *args)
        else:
            p = density_distribution(xi)
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

        return self

    def set_velocities(
        self,
        stellar_mass: float,
        gravitational_constant: float,
        q_index: float,
        aspect_ratio: float,
        reference_radius: float,
        rotation_axis: Union[Tuple[float, float, float], np.ndarray] = None,
        rotation_angle: float = None,
        pressureless: bool = False,
    ) -> Disc:
        """
        Set the disc particle velocities.

        Parameters
        ----------
        stellar_mass
            The mass of the central object the disc is orbiting.
        gravitational_constant
            The gravitational constant.
        q_index
            The index in the sound speed power law such that
                H ~ (R / R_reference) ^ (3/2 - q).
        aspect_ratio
            The aspect ratio at the reference radius.
        reference_radius
            The radius at which the aspect ratio is given.

        Optional Parameters
        -------------------
        rotation_axis
            An axis around which to rotate the disc.
        rotation_angle
            The angle to rotate around the rotation_axis.
        pressureless
            Set to True if the particles are pressureless, i.e. dust.
        """

        if self._positions is None:
            raise ValueError('Set positions first')

        if (rotation_axis is not None) ^ (rotation_angle is not None):
            raise ValueError(
                'Must specify rotation_angle and rotation_axis to perform rotation'
            )

        if rotation_axis is not None:
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation = spatial.transform.Rotation.from_rotvec(
                rotation_angle * rotation_axis
            )

        radius = np.sqrt(self._positions[:, 0] ** 2 + self._positions[:, 1] ** 2)
        phi = np.arctan2(self._positions[:, 1], self._positions[:, 0])

        omega = np.sqrt(gravitational_constant * stellar_mass / radius)

        if not pressureless:
            h_over_r = aspect_ratio * (radius / reference_radius) ** (3 / 2 - q_index)
            v_phi = omega * np.sqrt(1 - h_over_r ** 2)
        else:
            v_phi = omega

        v_z = np.zeros_like(radius)

        vxyz = np.array([-v_phi * np.sin(phi), v_phi * np.cos(phi), v_z]).T

        if rotation_axis is not None:
            vxyz = rotation.apply(vxyz)

        self._velocities = vxyz

        return self
