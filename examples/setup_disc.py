"""
Set up a protoplanetary disc around a star, and add a planet.
"""

import matplotlib.pyplot as plt
import numpy as np
import phantomsetup

# ------------------------------------------------------------------------------------ #
# Script options

plot_setup = True

# ------------------------------------------------------------------------------------ #
# Constants

igas = phantomsetup.defaults.particle_type['igas']

# ------------------------------------------------------------------------------------ #
# Parameters

prefix = 'disc'

number_of_particles = 1_000_000
particle_type = igas

alpha_artificial = 0.1

length_unit = phantomsetup.units.unit_string_to_cgs('au')
mass_unit = phantomsetup.units.unit_string_to_cgs('solarm')

radius_min = 10.0
radius_max = 200.0

disc_mass = 0.01

gravitational_constant = 1.0
stellar_mass = 1.0
stellar_accretion_radius = 5.0
stellar_position = (0.0, 0.0, 0.0)
stellar_velocity = (0.0, 0.0, 0.0)

ieos = 3
q_index = 0.75
aspect_ratio = 0.05
reference_radius = 10.0

radius_critical = 100.0
gamma = 1.0

planet_mass = 0.001
planet_position = (100.0, 0.0, 0.0)
planet_accretion_radius_fraction_hill_radius = 0.25

orbital_radius = np.sqrt(planet_position[0] ** 2 + planet_position[1] ** 2)
planet_hill_radius = phantomsetup.orbits.hill_sphere_radius(
    orbital_radius, planet_mass, stellar_mass
)
planet_accretion_radius = (
    planet_accretion_radius_fraction_hill_radius * planet_hill_radius
)
planet_velocity = np.sqrt(gravitational_constant * stellar_mass / orbital_radius)

# ------------------------------------------------------------------------------------ #
# Surface density distribution


def density_distribution(radius, radius_critical, gamma):
    """Self-similar disc surface density distribution.

    This is the Lyden-Bell and Pringle (1974) solution, i.e. a power law
    with an exponential taper.
    """
    return phantomsetup.disc.self_similar_accretion_disc(radius, radius_critical, gamma)


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

polyk = phantomsetup.eos.polyk_for_locally_isothermal_disc(
    q_index, reference_radius, aspect_ratio, stellar_mass, gravitational_constant
)

setup.set_equation_of_state(ieos=ieos, polyk=polyk)

# ------------------------------------------------------------------------------------ #
# Set viscosity to disc viscosity

setup.set_dissipation(disc_viscosity=True, alpha=alpha_artificial)

# ------------------------------------------------------------------------------------ #
# Add star

setup.add_sink(
    mass=stellar_mass,
    accretion_radius=stellar_accretion_radius,
    position=stellar_position,
    velocity=stellar_velocity,
)

# ------------------------------------------------------------------------------------ #
# Add disc

disc = phantomsetup.Disc()
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
    args=(radius_critical, gamma),
)
setup.add_disc(disc)

# ------------------------------------------------------------------------------------ #
# Add planet

setup.add_sink(
    mass=planet_mass,
    accretion_radius=planet_accretion_radius,
    position=planet_position,
    velocity=planet_velocity,
)

# ------------------------------------------------------------------------------------ #
# Write dump file and in file

setup.write_dump_file()
setup.write_in_file()

# ------------------------------------------------------------------------------------ #
# Plot some quantities

if plot_setup:

    fig, ax = plt.subplots()
    ax.plot(setup.x[::10], setup.y[::10], 'k.', ms=0.5)
    for sink in setup.sinks:
        ax.plot(sink.position[0], sink.position[1], 'ro')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')

    fig, ax = plt.subplots()
    ax.plot(setup.R[::10], setup.z[::10], 'k.', ms=0.5)
    ax.set_xlabel('$R$')
    ax.set_ylabel('$z$')
    ax.set_aspect('equal')
    ax.set_ylim(bottom=2 * setup.z.min(), top=2 * setup.z.max())

    fig, ax = plt.subplots()
    ax.plot(setup.R[::10], setup.vphi[::10], 'k.', ms=0.5)
    ax.set_xlabel('$R$')
    ax.set_ylabel('$v_{phi}$')

    plt.show()
