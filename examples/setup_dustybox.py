import numpy as np
import phantomsetup

# ------------------------------------------------------------------------------------ #
# Parameters
hfact = 1.0
prefix = 'test'

# ------------------------------------------------------------------------------------ #
# Instantiate phantomsetup object
setup = phantomsetup.Setup()

# ------------------------------------------------------------------------------------ #
# Setup box
box = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
vol = (box[1] - box[0]) * (box[3] - box[2]) * (box[5] - box[4])

# ------------------------------------------------------------------------------------ #
# Setup gas
particle_type = phantomsetup.defaults.igas
npartx = 32
rho = 1.0

# Particle positions
particle_spacing = (box[1] - box[0]) / npartx
position, smoothing_length = phantomsetup.dist.uniform_distribution(
    box_dimensions=box, particle_spacing=particle_spacing, hfact=hfact
)
npart = position.shape[0]
particle_mass = rho * vol / npart

# Particle velocities
velocity = np.zeros((npart, 3))

# Add particles
setup.add_particles(particle_type, particle_mass, position, velocity, smoothing_length)

# ------------------------------------------------------------------------------------ #
# Setup dust
particle_type = phantomsetup.defaults.idust
npartx = 16
rho = 0.01

# Particle positions
particle_spacing = (box[1] - box[0]) / npartx
position, smoothing_length = phantomsetup.dist.uniform_distribution(
    box_dimensions=box, particle_spacing=particle_spacing, hfact=hfact
)
npart = position.shape[0]
particle_mass = rho * vol / npart

# Particle velocities
velocity = np.hstack((np.ones((npart, 1)), np.zeros((npart, 2))))

# Add particles
setup.add_particles(particle_type, particle_mass, position, velocity, smoothing_length)

# ------------------------------------------------------------------------------------ #
# TODO: add other arrays...
alpha = np.zeros(setup.total_number_of_particles)
setup.add_array_to_particles('alpha', alpha)

# ------------------------------------------------------------------------------------ #
# TODO: add other header items...

# ------------------------------------------------------------------------------------ #
# Write dump
setup.write_dump_file()

# ------------------------------------------------------------------------------------ #
# DEBUGGING
if False:

    import matplotlib.pyplot as plt
    from IPython import get_ipython

    ipython = get_ipython()
    ipython.magic('matplotlib')

    fig, axes = plt.subplots(1, 3)

    axes[0].plot(position[:, 0], position[:, 1], '.')
    axes[0].set_aspect('equal')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    axes[1].plot(position[:, 0], position[:, 2], '.')
    axes[1].set_aspect('equal')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')

    axes[2].plot(position[:, 1], position[:, 2], '.')
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('z')

    plt.tight_layout()
    plt.show()
