"""Accretion disc."""

from __future__ import annotations

import functools
from typing import Callable, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy import integrate, spatial, stats

from . import constants
from .particles import Particles


class Disc(Particles):
    """Accretion disc.

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
    extra_args
        Extra arguments to pass to density_distribution.
    pressureless
        Set to True if the particles are pressureless, i.e. dust.

    Examples
    --------
    TODO: add examples
    """

    def __init__(
        self,
        *,
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
        rotation_axis: Union[Tuple[float, float, float], ndarray] = None,
        rotation_angle: float = None,
        extra_args: tuple = None,
        pressureless: bool = False,
    ):
        super().__init__()

        particle_mass = disc_mass / number_of_particles

        position, smoothing_length = self._set_positions(
            number_of_particles=number_of_particles,
            disc_mass=disc_mass,
            density_distribution=density_distribution,
            radius_range=radius_range,
            q_index=q_index,
            aspect_ratio=aspect_ratio,
            reference_radius=reference_radius,
            centre_of_mass=centre_of_mass,
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            extra_args=extra_args,
        )

        velocity = self._set_velocities(
            position=position,
            stellar_mass=stellar_mass,
            gravitational_constant=gravitational_constant,
            q_index=q_index,
            aspect_ratio=aspect_ratio,
            reference_radius=reference_radius,
            rotation_axis=rotation_axis,
            rotation_angle=rotation_angle,
            pressureless=pressureless,
        )

        self.add_particles(
            particle_type=particle_type,
            particle_mass=particle_mass,
            position=position,
            velocity=velocity,
            smoothing_length=smoothing_length,
        )

    def _set_positions(
        self,
        *,
        number_of_particles: float,
        disc_mass: float,
        density_distribution: Callable[[float], float],
        radius_range: Tuple[float, float],
        q_index: float,
        aspect_ratio: float,
        reference_radius: float,
        hfact: float = 1.2,
        centre_of_mass: Tuple[float, float, float] = None,
        rotation_axis: Union[Tuple[float, float, float], ndarray] = None,
        rotation_angle: float = None,
        extra_args: tuple = None,
    ) -> Tuple[ndarray, ndarray]:
        """Set the disc particle positions.

        Parameters
        ----------
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

        Optional Parameters
        -------------------
        hfact
            The smoothing length factor. Default is 1.2, as for cubic.
        centre_of_mass
            The centre of mass of the disc, i.e. around which position
            is it rotating.
        rotation_axis
            An axis around which to rotate the disc.
        rotation_angle
            The angle to rotate around the rotation_axis.
        extra_args
            Extra arguments to pass to density_distribution.

        Returns
        -------
        position : ndarray
            The particle positions.
        smoothing_length : ndarray
            The particle smoothing lengths.
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

        particle_mass = disc_mass / number_of_particles

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
        if extra_args is not None:
            p = density_distribution(xi, *extra_args)
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

        position = np.array([r * np.cos(phi), r * np.sin(phi), z]).T

        integrated_mass = integrate.quad(
            lambda x: 2 * np.pi * x * density_distribution(x, *extra_args),
            radius_range[0],
            radius_range[1],
        )[0]

        normalization = disc_mass / integrated_mass
        sigma = normalization * density_distribution(r, *extra_args)

        density = sigma * np.exp(-0.5 * (z / H) ** 2) / (H * np.sqrt(2 * np.pi))
        smoothing_length = hfact * (particle_mass / density) ** (1 / 3)

        if rotation_axis is not None:
            position = rotation.apply(position)

        position += centre_of_mass

        return position, smoothing_length

    def _set_velocities(
        self,
        *,
        position: ndarray,
        stellar_mass: float,
        gravitational_constant: float,
        q_index: float,
        aspect_ratio: float,
        reference_radius: float,
        rotation_axis: Union[Tuple[float, float, float], ndarray] = None,
        rotation_angle: float = None,
        pressureless: bool = False,
    ) -> ndarray:
        """Set the disc particle velocities.

        Parameters
        ----------
        position
            The particle positions.
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

        Returns
        -------
        velocity : ndarray
            The particle velocities.
        """
        if (rotation_axis is not None) ^ (rotation_angle is not None):
            raise ValueError(
                'Must specify rotation_angle and rotation_axis to perform rotation'
            )

        if rotation_axis is not None:
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation = spatial.transform.Rotation.from_rotvec(
                rotation_angle * rotation_axis
            )

        radius = np.sqrt(position[:, 0] ** 2 + position[:, 1] ** 2)
        phi = np.arctan2(position[:, 1], position[:, 0])

        omega = np.sqrt(gravitational_constant * stellar_mass / radius)

        if not pressureless:
            h_over_r = aspect_ratio * (radius / reference_radius) ** (1 / 2 - q_index)
            v_phi = omega * np.sqrt(1 - h_over_r ** 2)
        else:
            v_phi = omega

        v_z = np.zeros_like(radius)

        velocity = np.array([-v_phi * np.sin(phi), v_phi * np.cos(phi), v_z]).T

        if rotation_axis is not None:
            velocity = rotation.apply(velocity)

        return velocity


def smoothing_length_on_scale_height(
    radius: ndarray,
    smoothing_length: ndarray,
    reference_radius: float,
    aspect_ratio: float,
    q_index: float,
    sample_number: int = None,
):
    """Calculate the average smoothing length on scale height.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    if sample_number is None:
        bins = 10
    else:
        bins = sample_number

    binned_rh = stats.binned_statistic(radius, smoothing_length, bins=bins)

    r = binned_rh.bin_edges
    r = r[:-1] + (r[1:] - r[:-1]) / 2

    h = binned_rh.statistic

    H = reference_radius ** (q_index - 1 / 2) * aspect_ratio * r ** (3 / 2 - q_index)

    return h / H


def keplerian_angular_velocity(
    radius: Union[float, ndarray],
    mass: Union[float, ndarray],
    gravitational_constant: float = None,
) -> Union[float, ndarray]:
    """Keplerian angular velocity Omega.

    Parameters
    ----------
    radius
        The distance from the central object.
    mass
        The central object mass.
    gravitational_constant
        The gravitational constant in appropriate units.
    """
    if gravitational_constant is None:
        gravitational_constant = constants.gravitational_constant
    return np.sqrt(gravitational_constant * mass / radius ** 3)


def add_gap(
    orbital_radius: float, gap_width: float
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """Decorate by adding a gap in a density distribution.

    The gap is a step function. I.e. the density is zero within the gap.

    Parameters
    ----------
    radius_planet
        The planet radius.
    gap_width
        The gap width centered on the planet.

    Returns
    -------
    callable
        The density distribution with the added gap.
    """

    def wrapper_outer(distribution):
        @functools.wraps(distribution)
        def wrapper_inner(radius, *extra_args):

            result = distribution(radius, *extra_args)
            gap_radii = np.logical_and(
                orbital_radius - 0.5 * gap_width < radius,
                radius < orbital_radius + 0.5 * gap_width,
            )

            if isinstance(result, ndarray):
                result[gap_radii] = 0.0
            elif gap_radii:
                result = 0.0

            return result

        return wrapper_inner

    return wrapper_outer


def power_law(
    radius: Union[float, ndarray], reference_radius: float, p_index: float
) -> Union[float, ndarray]:
    """Power law distribution.

    (R / R_ref)^(-p)

    Parameters
    ----------
    radius
        The radius at which to evaulate the function.
    reference_radius
        The reference radius
    p_index
        The exponent in the power law.

    Returns
    -------
    float
        The surface density at the specified radius.
    """
    ref, p = reference_radius, p_index
    return (radius / ref) ** (-p)


def power_law_with_zero_inner_boundary(
    radius: Union[float, ndarray],
    inner_radius: float,
    reference_radius: float,
    p_index: float,
) -> Union[float, ndarray]:
    """Power law distribution with zero inner boundary condition.

    (R / R_ref)^(-p) * [1 - sqrt(R_inner / R)]

    Parameters
    ----------
    radius
        The radius at which to evaulate the function.
    inner_radius
        The inner radius.
    reference_radius
        The reference radius.
    p_index
        The exponent in the power law.

    Returns
    -------
    float
        The surface density at the specified radius.
    """
    ref, inner, p = reference_radius, inner_radius, p_index
    return (radius / ref) ** (-p) * (1 - np.sqrt(inner / radius))


def self_similar_accretion_disc(
    radius: Union[float, ndarray], radius_critical: float, gamma: float
) -> Union[float, ndarray]:
    """Lynden-Bell and Pringle (1974) self-similar solution.

    (R / R_crit)^(-y) * exp[-(R / R_crit) ^ (2 - y)]

    Parameters
    ----------
    radius
        The radius at which to evaulate the function.
    radius_critical
        The critical radius for the exponential taper.
    gamma
        The exponent in the power law.

    Returns
    -------
    float
        The surface density at the specified radius.
    """
    rc, y = radius_critical, gamma
    return (radius / rc) ** (-y) * np.exp(-((radius / rc) ** (2 - y)))


def self_similar_accretion_disc_with_zero_inner_boundary(
    radius: Union[float, ndarray],
    radius_inner: float,
    radius_critical: float,
    gamma: float,
) -> Union[float, ndarray]:
    """Self-similar solution with a zero inner boundary condition.

    (R / R_crit)^(-y) * exp[-(R / R_crit) ^ (2 - y)]

    Parameters
    ----------
    radius
        The radius at which to evaulate the function.
    inner_radius
        The inner radius.
    radius_critical
        The critical radius for the exponential taper.
    gamma
        The exponent in the power law.

    Returns
    -------
    float
        The surface density at the specified radius.
    """
    inner, rc, y = radius_inner, radius_critical, gamma
    return (
        (radius / rc) ** (-y)
        * np.exp(-((radius / rc) ** (2 - y)))
        * (1 - np.sqrt(inner / radius))
    )
