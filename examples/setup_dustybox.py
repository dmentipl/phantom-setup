"""
Set up the dusty box problem.

The dust and gas are co-located in a box with uniform density. There is
an initial uniform differential velocity between the dust and gas.
"""

import numpy as np
import phantomsetup

# ------------------------------------------------------------------------------------ #
# Constants

igas = phantomsetup.defaults.particle_type['igas']
idust = phantomsetup.defaults.particle_type['idust']

# ------------------------------------------------------------------------------------ #
# Parameters

# The file name prefix, such that the dump file is prefix_00000.tmp.h5 and the
# in file is prefix.in.
prefix = 'dustybox'

# hfact should be 1.0 for the quintic kernel which is the appropriate
# kernel for calculations with dust
hfact = 1.0

# ieos of 1 sets the isothermal equation of state
ieos = 1

# The boundary of the box as (xmin, xmax, ymin, ymax, zmin, zmax).
box_boundary = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

# The isothermal sound speed.
sound_speed = 1.0

# The number of gas particles.
number_of_particles_gas = 50_000

# The number of dust particles in each species.
number_of_particles_dust = 10_000

# The initial uniform density of the gas.
density_gas = 1.0

# The dust-to-gas ratio for each dust species.
dust_to_gas_ratio = (0.01, 0.02, 0.03, 0.04, 0.05)

# The dust drag method. Options are "Epstein/Stokes", "K_const", or "ts_const".
drag_method = 'K_const'

# The constant drag coefficient.
K_code = 1.0

# The grain size of each dust species.
grain_size = ()

# The intrinsic grain density.
grain_density = 3.0

# The initial delta in uniform velocity between gas and dust.
velocity_delta = 1.0

# ------------------------------------------------------------------------------------ #
# Instantiate phantomsetup object

setup = phantomsetup.Setup()

# ------------------------------------------------------------------------------------ #
# File prefix

setup.prefix = prefix

# ------------------------------------------------------------------------------------ #
# Set units

length_unit = phantomsetup.units.unit_string_to_cgs('au')
mass_unit = phantomsetup.units.unit_string_to_cgs('solarm')
time_unit = phantomsetup.units.unit_string_to_cgs('year')

setup.set_units(length=length_unit, mass=mass_unit, time=time_unit)

# ------------------------------------------------------------------------------------ #
# Set equation of state

setup.set_equation_of_state(ieos=ieos, polyk=sound_speed ** 2)

# ------------------------------------------------------------------------------------ #
# Set dust

number_of_dust_species = len(dust_to_gas_ratio)

density_dust = [eps * density_gas for eps in dust_to_gas_ratio]

if drag_method == 'Epstein/Stokes':
    setup.set_dust(
        dust_method='largegrains',
        drag_method=drag_method,
        grain_size=grain_size,
        grain_density=grain_density,
    )

elif drag_method in ('K_const', 'ts_const'):
    setup.set_dust(
        dust_method='largegrains',
        drag_method=drag_method,
        drag_constant=K_code,
        number_of_dust_species=number_of_dust_species,
    )

# ------------------------------------------------------------------------------------ #
# Set boundary

setup.set_boundary(box_boundary)

# ------------------------------------------------------------------------------------ #
# Add gas particles to box


def velocity_distribution(xyz: np.ndarray) -> np.ndarray:
    """Gas has zero initial velocity."""
    vxyz = np.zeros_like(xyz)
    return vxyz


box = phantomsetup.Box(*box_boundary)
box.add_particles(
    particle_type=igas,
    number_of_particles=number_of_particles_gas,
    density=density_gas,
    velocity_distribution=velocity_distribution,
    hfact=hfact,
)
setup.add_box(box)

# ------------------------------------------------------------------------------------ #
# Add dust particles to box


def velocity_distribution(xyz: np.ndarray) -> np.ndarray:
    """Dust has uniform initial velocity."""
    vxyz = np.zeros_like(xyz)
    vxyz[:, 0] = velocity_delta
    return vxyz


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

# ------------------------------------------------------------------------------------ #
# Add extra quantities to particles

alpha = np.zeros(setup.total_number_of_particles, dtype=np.single)
setup.add_array_to_particles('alpha', alpha)

# ------------------------------------------------------------------------------------ #
# Write dump file and in file

setup.write_dump_file()
setup.write_in_file()
