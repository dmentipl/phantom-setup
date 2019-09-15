"""Demonstrate use phantomsetup.disc.Disc."""

import matplotlib.pyplot as plt
import numpy as np
import phantomsetup
from phantomsetup import defaults
from phantomsetup.disc import Disc
from phantomsetup.units import unit_string_to_cgs
from phantomsetup.utils import (
    accretion_disc_self_similar,
    add_gap,
    polyk_for_locally_isothermal_disc,
)

# ------------------------------------------------------------------------------------ #
# Constants

igas = defaults.particle_type['igas']

# ------------------------------------------------------------------------------------ #
# Parameters

prefix = 'disc'

number_of_particles = 1_000_000
particle_type = igas

length_unit = unit_string_to_cgs('au')
mass_unit = unit_string_to_cgs('solarm')

radius_min = 10.0
radius_max = 200.0

disc_mass = 0.01

gravitational_constant = 1.0
stellar_mass = 1.0
stellar_radius = 5.0
stellar_position = (0.0, 0.0, 0.0)
stellar_velocity = (0.0, 0.0, 0.0)

ieos = 3
q_index = 0.75
aspect_ratio = 0.05
reference_radius = 10.0

radius_critical = 100.0
gamma = 1.0

rotation_angle = np.pi / 3
rotation_axis = (1, 1, 0)

radius_planet = 100.0
gap_width = 10.0


# ------------------------------------------------------------------------------------ #
# Surface density distribution

# We use the Lynden-Bell and Pringle (1974) self-similar solution, i.e.
# a power law with an exponential taper. Plus we add a gap of 10 au at
# 100 au.


@add_gap(radius_planet=radius_planet, gap_width=gap_width)
def density_distribution(radius, radius_critical, gamma):
    """Surface density distribution.

    Self-similar disc solution with a gap added.
    """
    return accretion_disc_self_similar(radius, radius_critical, gamma)


args = (radius_critical, gamma)

# ------------------------------------------------------------------------------------ #
# Instantiate Setup object

setup = phantomsetup.Setup()
setup.prefix = prefix

# ------------------------------------------------------------------------------------ #
# Set units

setup.set_units(
    length=length_unit, mass=mass_unit, gravitational_constant_is_unity=True
)

# ------------------------------------------------------------------------------------ #
# Set equation of state

polyk = polyk_for_locally_isothermal_disc(
    q_index, reference_radius, aspect_ratio, stellar_mass, gravitational_constant
)

setup.set_equation_of_state(ieos=ieos, polyk=polyk)

# ------------------------------------------------------------------------------------ #
# Add star

setup.add_sink(
    mass=stellar_mass,
    accretion_radius=stellar_radius,
    position=stellar_position,
    velocity=stellar_velocity,
)

# ------------------------------------------------------------------------------------ #
# Instantiate Disc object

disc = Disc()

# ------------------------------------------------------------------------------------ #
# Add particles to disc

disc.add_particles(
    particle_type=particle_type,
    number_of_particles=number_of_particles,
    disc_mass=disc_mass,
    density_distribution=density_distribution,
    radius_range=(radius_min, radius_max),
    q_index=q_index,
    aspect_ratio=aspect_ratio,
    reference_radius=reference_radius,
    stellar_mass=stellar_mass,
    gravitational_constant=gravitational_constant,
    rotation_axis=rotation_axis,
    rotation_angle=rotation_angle,
    args=(radius_critical, gamma),
)

# ------------------------------------------------------------------------------------ #
# Add particles from disc to setup

setup.add_particles(
    particle_type=disc.particle_type,
    particle_mass=disc.particle_mass,
    positions=disc.positions,
    velocities=disc.velocities,
    smoothing_length=disc.smoothing_length,
)

# ------------------------------------------------------------------------------------ #
# Write dump file and in file
setup.write_dump_file()
setup.write_in_file()

# ------------------------------------------------------------------------------------ #
# Plot some quantities

x, y, z = disc.positions[:, 0], disc.positions[:, 1], disc.positions[:, 2]
R = np.sqrt(x ** 2 + y ** 2)

vx, vy, vz = disc.velocities[:, 0], disc.velocities[:, 1], disc.velocities[:, 2]
vphi = np.sqrt(vx ** 2 + vy ** 2)

fig, ax = plt.subplots(1, 2)
ax[0].plot(x[::10], y[::10], '.')
ax[0].set_aspect('equal')
ax[1].plot(R[::10], z[::10], '.')
ax[1].set_aspect('equal')

fig, ax = plt.subplots()
ax.plot(R[::10], vphi[::10], '.')

plt.show()
