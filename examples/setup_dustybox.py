import numpy as np
import phantomsetup.particle_distributions as dist
import phantomsetup.phantomsetup as ps

# ------------------------------------------------------------------------------------ #
# Constants
IGAS = 1
IDUST = 7

# ------------------------------------------------------------------------------------ #
# Parameters
hfact = 1.0
time = 0.0
hfact = 1.0
gamma = 1.0
tmax = 0.1
dtmax = 0.001
cs = 1.0
polyk = cs ** 2

# ------------------------------------------------------------------------------------ #
# Instantiate phantomsetup object
setup = ps.Setup('dustybox')

# ------------------------------------------------------------------------------------ #
# Setup box
box = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
vol = (box[1] - box[0]) * (box[3] - box[2]) * (box[5] - box[4])

# ------------------------------------------------------------------------------------ #
# Setup gas
itype = IGAS
npartx = 32
rho = 1.0

# Particle positions
particle_spacing = (box[1] - box[0]) / npartx
position, smoothing_length = dist.uniform_distribution(
    box_dimensions=box, particle_spacing=particle_spacing, hfact=hfact
)
npart = position.shape[0]
particle_mass = rho * vol / npart

# Particle velocities
velocity = np.zeros((npart, 3))
velocity[:, 0] = 0.0

# Add particles
setup.add_particles(itype, position, velocity, smoothing_length)
setup.particle_mass.update({itype: particle_mass})

# ------------------------------------------------------------------------------------ #
# Setup dust
itype = IDUST
npartx = 16
rho = 0.01

# Particle positions
particle_spacing = (box[1] - box[0]) / npartx
position, smoothing_length = dist.uniform_distribution(
    box_dimensions=box, particle_spacing=particle_spacing, hfact=hfact
)
npart = position.shape[0]
particle_mass = rho * vol / npart

# Particle velocities
velocity = np.zeros((npart, 3))
velocity[:, 0] = 1.0

# Add particles
setup.add_particles(itype, position, velocity, smoothing_length)
setup.particle_mass.update({IGAS: particle_mass})
setup.particle_mass.update({itype: particle_mass})

# ------------------------------------------------------------------------------------ #
# TODO: add other arrays...
setup.add_array_to_particles('alpha', np.zeros(setup.total_number_of_particles))

# ------------------------------------------------------------------------------------ #
# TODO: add other header items...
setup.fileident = 'fulldump: Phantom 1.3.0 6666c55 (hydro+dust): 28/08/2019 19:21:16.8'

# ------------------------------------------------------------------------------------ #
# Write dump
setup.write_dump_file('test_00000.tmp.h5')

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
