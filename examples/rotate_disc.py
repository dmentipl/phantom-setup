"""Demonstrate use phantomsetup.disc.Disc."""

import matplotlib.pyplot as plt
import numpy as np
from phantomsetup.disc import Disc

radius_min = 10.0
radius_max = 200.0
n_particles = int(1e6)

q_index = 0.75
aspect_ratio = 0.05
reference_radius = 10.0

radius_critical = 100.0
gamma = 1.0


def density_distribution(radius, radius_critical, gamma):
    rc, y = radius_critical, gamma
    return (radius / rc) * (-y) * np.exp(-(radius / rc) ** (2 - y))


disc = Disc()

disc.set_positions(
    number_of_particles=n_particles,
    density_distribution=density_distribution,
    radius_range=(radius_min, radius_max),
    q_index=q_index,
    aspect_ratio=aspect_ratio,
    reference_radius=reference_radius,
    rotation_angle=np.pi / 3,
    rotation_axis=(1, 1, 0),
    args=(radius_critical, gamma),
)

x, y, z = disc.positions[:, 0], disc.positions[:, 1], disc.positions[:, 2]
R = np.sqrt(x ** 2 + y ** 2)

fig, ax = plt.subplots(1, 2)
ax[0].plot(x[::10], y[::10], '.')
ax[0].set_aspect('equal')
ax[1].plot(R[::10], z[::10], '.')
ax[1].set_aspect('equal')

plt.show()
