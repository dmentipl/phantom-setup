#%% [markdown]
# # Dusty box
#
# This notebook demonstrates setting up the dusty box test.
#
# The dust and gas are co-located in a box with uniform density. There is an initial uniform differential velocity between the dust and gas.
#
# ## Initialization
#
# Import required modules.

#%%
import numpy as np
import phantomsetup

#%% [markdown]
# Set constants.

#%%
igas = phantomsetup.defaults.particle_type['igas']
idust = phantomsetup.defaults.particle_type['idust']

#%% [markdown]
# ## Parameters
#
# We set the parameters for the setup.
#
# ### Prefix
#
# We set the file name prefix, such that the dump file is `prefix_00000.tmp.h5` and the in file is `prefix.in`.

#%%
prefix = 'dustybox'

#%% [markdown]
# ### `hfact`
#
# The smoothing length factor `hfact` should be 1.0 for the quintic kernel which is the appropriate kernel for calculations with dust.

#%%
hfact = 1.0

#%% [markdown]
# ### Equation of state
#
# An `ieos` of 1 sets the globally isothermal equation of state. The sound speed is the only free parameter.

#%%
ieos = 1
sound_speed = 1.0

#%% [markdown]
# ### Boundary
#
# The boundary of the box as (xmin, xmax, ymin, ymax, zmin, zmax).

#%%
box_boundary = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

#%% [markdown]
# ### Resolution
#
# The number of gas particles.

#%%
number_of_particles_gas = 50_000

#%% [markdown]
# The number of dust particles in each species.

#%%
number_of_particles_dust = 10_000

#%% [markdown]
# ### Density
#
# The initial uniform density of the gas.

#%%
density_gas = 1.0

#%% [markdown]
# The dust-to-gas ratio for each dust species.

#%%
dust_to_gas_ratio = (0.01, 0.02, 0.03, 0.04, 0.05)

#%% [markdown]
# ### Dust
#
# The dust drag method. Options are "Epstein/Stokes", "K_const", or "ts_const".

#%%
drag_method = 'K_const'

#%% [markdown]
# The constant drag coefficient. This is not required for Epstein/Stokes drag.

#%%
K_code = 1.0

#%% [markdown]
# The grain size of each dust species. We don't set this for constant drag as it is not required.

#%%
grain_size = ()

#%% [markdown]
# The intrinsic grain density.

#%%
grain_density = 3.0

#%% [markdown]
# ### Velocity
#
# The initial delta in uniform velocity between gas and dust.

#%%
velocity_delta = 1.0

#%% [markdown]
# ## Instantiate phantomsetup object
#
# We instantiate the `phantom.Setup` object.

#%%
setup = phantomsetup.Setup()

#%% [markdown]
# Then we add the previously defined parameters to this object.
#
# ### File prefix

#%%
setup.prefix = prefix

#%% [markdown]
# ### Set units
#
# We convert unit strings to cgs values to pass to the `set_units` method.

#%%
length_unit = phantomsetup.units.unit_string_to_cgs('au')
mass_unit = phantomsetup.units.unit_string_to_cgs('solarm')
time_unit = phantomsetup.units.unit_string_to_cgs('year')

#%%
setup.set_units(length=length_unit, mass=mass_unit, time=time_unit)

#%% [markdown]
# ### Set equation of state

#%%
setup.set_equation_of_state(ieos=ieos, polyk=sound_speed ** 2)

#%% [markdown]
# ### Set dust
#
# Here we call the `set_dust` method differently depending on the drag type. First we get the number of species from the `dust_to_gas_ratio` tuple.

#%%
number_of_dust_species = len(dust_to_gas_ratio)

#%% [markdown]
# And we set the dust density from the dust-to-gas ratio.

#%%
density_dust = [eps * density_gas for eps in dust_to_gas_ratio]

#%% [markdown]
# Then we initialize the dust via the `set_dust` method.

#%%
if drag_method == 'Epstein/Stokes':
    setup.set_dust(
        dust_method='largegrains',
        drag_method=drag_method,
        grain_size=grain_size,
        grain_density=grain_density,
    )

    #%%
elif drag_method in ('K_const', 'ts_const'):
    setup.set_dust(
        dust_method='largegrains',
        drag_method=drag_method,
        drag_constant=K_code,
        number_of_dust_species=number_of_dust_species,
    )

#%% [markdown]
# ### Set boundary
#
# This sets the boundary, and the boundary conditions to periodic.

#%%
setup.set_boundary(box_boundary, periodic=True)

#%% [markdown]
# ### Make a box of particles
#
# The `Box` class sets up a box of particles in a uniform spatial distribution with an arbitrary velocity field.
#
# #### Make a gas box
#
# We first define a velocity distribution. Here the gas velocity is zero everywhere.

#%%
def velocity_distribution(xyz: np.ndarray) -> np.ndarray:
    """Gas has zero initial velocity."""
    vxyz = np.zeros_like(xyz)
    return vxyz

#%% [markdown]
# Then we instantiate a `Box` object, add particles, and add it to the setup.

#%%
box = phantomsetup.Box(*box_boundary)
box.add_particles(
    particle_type=igas,
    number_of_particles=number_of_particles_gas,
    density=density_gas,
    velocity_distribution=velocity_distribution,
    hfact=hfact,
)
setup.add_box(box)

#%% [markdown]
# #### Make dust boxes
#
# We do the same for each dust species. First, the velocity distribution. Here the dust velocity is uniform and in the x-direction.

#%%
def velocity_distribution(xyz: np.ndarray) -> np.ndarray:
    """Dust has uniform initial velocity."""
    vxyz = np.zeros_like(xyz)
    vxyz[:, 0] = velocity_delta
    return vxyz

#%% [markdown]
# Then we iterate over each of the dust species.

#%%
for idx in range(number_of_dust_species):
    box = phantomsetup.Box(*box_boundary)
    box.add_particles(
        particle_type=idust + idx,
        number_of_particles=number_of_particles_dust,
        density=density_dust[idx],
        velocity_distribution=velocity_distribution,
        hfact=hfact,
    )
    setup.add_box(box)

#%% [markdown]
# ### Add extra quantities to particles
#
# Phantom requires the $\alpha$ viscosity parameter array. We set it to zero. Note that the array is single precision.

#%%
alpha = np.zeros(setup.total_number_of_particles, dtype=np.single)
setup.add_array_to_particles('alpha', alpha)

#%% [markdown]
# ## Write to file
#
# Now that we are happy with the setup, write the "temporary" dump file with the initial conditions and the Phantom "in" file.
#
# First we set a working directory for the simulation.

#%%
working_dir = '~/runs/dustybox'

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
