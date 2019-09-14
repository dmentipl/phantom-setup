"""Demonstrate use phantomsetup.disc.Disc."""

import matplotlib.pyplot as plt
import numpy as np
from phantomsetup.disc import Disc
from phantomsetup import defaults

# ------------------------------------------------------------------------------------ #
# Constants

igas = defaults.particle_type['igas']

# ------------------------------------------------------------------------------------ #
# Parameters

number_of_particles = 1_000_000
particle_type = igas

radius_min = 10.0
radius_max = 200.0

disc_mass = 0.01

q_index = 0.75
aspect_ratio = 0.05
reference_radius = 10.0

radius_critical = 100.0
gamma = 1.0

rotation_angle = np.pi / 3
rotation_axis = (1, 1, 0)

stellar_mass = 1.0
gravitational_constant = 1.0


def density_distribution(radius, radius_critical, gamma):
    rc, y = radius_critical, gamma
    return (radius / rc) ** (-y) * np.exp(-(radius / rc) ** (2 - y))


# ------------------------------------------------------------------------------------ #
# Instantiate Disc object

disc = Disc()

# ------------------------------------------------------------------------------------ #
# Set add particles to disc

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
