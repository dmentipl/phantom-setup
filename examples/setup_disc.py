#%% [markdown]
# Set up a disc
#
# In this tutorial we set up a protoplanetary disc around a star represented by a sink particle, and we add a planet. This notebook generates a Phantom "temporary" dump file that can be read by Phantom as an initial condition. It also generates a Phantom "in" file. Together, these files can start a Phantom simulation.
#
# ## Initialization
#
# First we import the required modules.

#%%
import matplotlib.pyplot as plt
import numpy as np
import phantomsetup

#%% [markdown]
# Here we set some constants for convenience.

#%%
igas = phantomsetup.defaults.particle_type['igas']

#%% [markdown]
# ## Parameters
#
# Now we set the parameters for the problem.
#
# First is the `prefix` which sets the file name for the dump file and Phantom in file.

#%%
prefix = 'disc'

#%% [markdown]
# ### Resolution
#
# We choose the resolution to be $10^6$ gas particles.

#%%
number_of_particles = 1_000_000
particle_type = igas

#%% [markdown]
# ### Viscosity
#
# The SPH $\alpha$ viscosity parameter is its minimal value of 0.1.

#%%
alpha_artificial = 0.1

#%% [markdown]
# ### Units
#
# We set the length and mass units to be au and solar masses, respectively. We will also set the time unit such that the gravitational constant is unity.

#%%
length_unit = phantomsetup.units.unit_string_to_cgs('au')
mass_unit = phantomsetup.units.unit_string_to_cgs('solarm')
gravitational_constant = 1.0

#%% [markdown]
# ### Star
#
# The star is of solar mass, at the origin, with a 5 au accretion radius.

#%%
stellar_mass = 1.0
stellar_accretion_radius = 5.0
stellar_position = (0.0, 0.0, 0.0)
stellar_velocity = (0.0, 0.0, 0.0)

#%% [markdown]
# ### Disc
#
# The disc has mass 0.01 solar masses, it extends from 10 au to 200 au.

#%%
radius_min = 10.0
radius_max = 200.0

#%%
disc_mass = 0.01

#%% [markdown]
# ### Equation of state
#
# The equation of state is locally isothermal. We set the aspect ratio H/R at a reference radius.

#%%
ieos = 3
q_index = 0.75
aspect_ratio = 0.05
reference_radius = 10.0

#%% [markdown]
# ### Planet
#
# We add a planet at 100 au.

#%%
planet_mass = 0.001
planet_position = (100.0, 0.0, 0.0)

#%%
orbital_radius = np.linalg.norm(planet_position)
planet_velocity = np.sqrt(gravitational_constant * stellar_mass / orbital_radius)

#%% [markdown]
# We set the planet accretion radius as a fraction of the Hill sphere radius.

#%%
planet_accretion_radius_fraction_hill_radius = 0.25

#%%
planet_hill_radius = phantomsetup.orbits.hill_sphere_radius(
    orbital_radius, planet_mass, stellar_mass
)
planet_accretion_radius = (
    planet_accretion_radius_fraction_hill_radius * planet_hill_radius
)

#%% [markdown]
# ### Surface density distribution
#
# For the surface density distribution we use the Lynden-Bell and Pringle (1974) self-similar solution, i.e. a power law with an exponential taper.

#%%
def density_distribution(radius, radius_critical, gamma):
    """Self-similar disc surface density distribution.

    This is the Lyden-Bell and Pringle (1974) solution, i.e. a power law
    with an exponential taper.
    """
    return phantomsetup.disc.self_similar_accretion_disc(radius, radius_critical, gamma)

radius_critical = 100.0
gamma = 1.0

#%%
args = (radius_critical, gamma)

#%% [markdown]
# ## Instantiate the `Setup` object
#
# The following instantiates the `phantomsetup.Setup` object.

#%%
setup = phantomsetup.Setup()

#%% [markdown]
# ## Set attributes and add particles
#
# ### Prefix
#
# Set the prefix.

#%%
setup.prefix = prefix

#%% [markdown]
# ### Units
#
# Set units.

#%%
setup.set_units(
    length=length_unit, mass=mass_unit, gravitational_constant_is_unity=True
)

#%% [markdown]
# ### Equation of state
#
# Set the equation of state. We get `polyk` from the aspect ratio parametrization.

#%%
polyk = phantomsetup.eos.polyk_for_locally_isothermal_disc(
    q_index, reference_radius, aspect_ratio, stellar_mass, gravitational_constant
)

#%%
setup.set_equation_of_state(ieos=ieos, polyk=polyk)

#%% [markdown]
# ### Viscosity
#
# Set the numerical viscosity to Phantom disc viscosity.

#%%
setup.set_dissipation(disc_viscosity=True, alpha=alpha_artificial)

#%% [markdown]
# ### Star
#
# Add a star at the origin.

#%%
setup.add_sink(
    mass=stellar_mass,
    accretion_radius=stellar_accretion_radius,
    position=stellar_position,
    velocity=stellar_velocity,
)

#%% [markdown]
# ### Disc
#
# Add the disc around the star.

#%%
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

#%% [markdown]
# ### Planet
#
# Add a planet in orbit around the star.

#%%
setup.add_sink(
    mass=planet_mass,
    accretion_radius=planet_accretion_radius,
    position=planet_position,
    velocity=planet_velocity,
)

#%% [markdown]
# ## Plot
#
# Now we plot some quantities to see what we have set up.
#
# First is the particles in the xy-plane. The sink particles are marked in red.

#%%
fig, ax = plt.subplots()
ax.plot(setup.x[::10], setup.y[::10], 'k.', ms=0.5)
for sink in setup.sinks:
    ax.plot(sink.position[0], sink.position[1], 'ro')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal')

#%% [markdown]
# Next we plot the particles in the rz-plane.

#%%
fig, ax = plt.subplots()
ax.plot(setup.R[::10], setup.z[::10], 'k.', ms=0.5)
ax.set_xlabel('$R$')
ax.set_ylabel('$z$')
ax.set_aspect('equal')
ax.set_ylim(bottom=2 * setup.z.min(), top=2 * setup.z.max())

#%% [markdown]
# Finally, we plot $v_{\phi}$ as a function of radius.

#%%
fig, ax = plt.subplots()
ax.plot(setup.R[::10], setup.vphi[::10], 'k.', ms=0.5)
ax.set_xlabel('$R$')
ax.set_ylabel('$v_{\phi}$')

#%% [markdown]
# ## Write to file
#
# Now that we are happy with the setup, write the "temporary" dump file with the initial conditions and the Phantom "in" file.
#
# First we set a working directory for the simulation.

#%%
working_dir = '~/runs/disc'

#%%
setup.write_dump_file(directory=working_dir)
setup.write_in_file(directory=working_dir)

#%% [markdown]
# ## Compile Phantom
#
# You can start a Phantom calculation from these two files but you must compile Phantom with the correct Makefile variables. We can use the `phantom_compile_command` method to show how Phantom would be compiled.

#%%
print(setup.phantom_compile_command())

#%% [markdown]
# We use the `compile_phantom` method to compile Phantom.

#%%
result = setup.compile_phantom(phantom_dir='~/repos/phantom', working_dir=working_dir)
