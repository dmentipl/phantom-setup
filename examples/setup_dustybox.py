"""
Setup the DUSTYBOX test problem.

The dust, with possibly multiple species, and gas are co-located in a
box. The gas is initially stationary, and the dust is given a uniform
velocity to the right.

This example instantiates a phantomsetup.Setup object with the name
'dustybox'. It uses the following methods:

    - set_units
    - set_boundary
    - set_equation_of_state
    - set_dust
    - add_particles
    - add_array_to_particles
    - write_dump_file
    - write_in_file

The main features are to set some parameters, add gas and dust
particles, add additional arrays, write a 'temporary' dump file (the
Phantom initial condition), and write a Phantom 'in' file.
"""

from typing import Collection, Union

import numpy as np
import phantomsetup

# TODO: things to set that are not currently set
# - units (if using physical drag, i.e. Epstein/Stokes)
# - grain_size and grain_density (if using physical drag, i.e. Epstein/Stokes)

# -------------------------------------------------------------------------------- #
# Constants

igas = phantomsetup.defaults.particle_type['igas']
idust = phantomsetup.defaults.particle_type['idust']

au = phantomsetup.constants.au
solarm = phantomsetup.constants.solarm
year = phantomsetup.constants.year / (2 * np.pi)


def main():

    # -------------------------------------------------------------------------------- #
    # Parameters

    prefix = 'dustybox'

    box_boundary = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

    hfact = 1.0
    ieos = 1
    sound_speed = 1.0

    number_of_particles_in_x_gas = 32
    number_of_particles_in_x_dust = 16

    density_gas = 1.0
    dust_to_gas_ratio = (0.01, 0.02, 0.03, 0.04, 0.05)

    drag_method = 'K_const'

    K_code = 1.0

    velocity_x_gas = 0.0
    velocity_x_dust = 1.0

    # -------------------------------------------------------------------------------- #
    # Instantiate phantomsetup object

    dustybox = phantomsetup.Setup()

    # -------------------------------------------------------------------------------- #
    # File prefix

    dustybox.prefix = prefix

    # -------------------------------------------------------------------------------- #
    # Set unit

    dustybox.set_units(au, solarm, year)

    # -------------------------------------------------------------------------------- #
    # Set equation of state

    dustybox.set_equation_of_state(ieos=ieos, polyk=sound_speed ** 2)

    # -------------------------------------------------------------------------------- #
    # Set dust grain size distribution

    dustybox.set_dust(
        dust_method='largegrains',
        drag_method=drag_method,
        drag_constant=K_code,
        number_of_dust_species=len(dust_to_gas_ratio),
    )

    # -------------------------------------------------------------------------------- #
    # Setup box

    dustybox.set_boundary(box_boundary)

    # -------------------------------------------------------------------------------- #
    # Add dust and gas particles to box

    add_particles_to_box(
        dustybox,
        number_of_particles_in_x_gas,
        number_of_particles_in_x_dust,
        density_gas,
        dust_to_gas_ratio,
        velocity_x_gas,
        velocity_x_dust,
        hfact,
    )

    # -------------------------------------------------------------------------------- #
    # Add extra quantities to particles

    alpha = np.zeros(dustybox.total_number_of_particles, dtype=np.single)
    dustybox.add_array_to_particles('alpha', alpha)

    # -------------------------------------------------------------------------------- #
    # Write dump and in file

    dustybox.write_dump_file()
    dustybox.write_in_file()

    # -------------------------------------------------------------------------------- #
    # Return the phantomsetup.Setup object

    return dustybox


def add_particles_to_box(
    setup: phantomsetup.Setup,
    number_of_particles_in_x_gas: int,
    number_of_particles_in_x_dust: int,
    density_gas: float,
    dust_to_gas_ratio: Union[Collection, np.ndarray],
    velocity_x_gas: float,
    velocity_x_dust: float,
    hfact: float,
):
    """
    Helper function to add particles to the box.

    Parameters
    ----------
    setup : phantomsetup.Setup
        The phantomsetup object representing the simulation.
    number_of_particles_in_x_gas: int
        The number of gas particles in the x-direction. Thus the total
        number of particles for a cube box would be roughly npartx**3.
    number_of_particles_in_x_dust: int
        The number of dust particles in the x-direction. Thus the total
        number of particles for a cube box would be roughly npartx**3.
    density_gas: float
        The initial gas density.
    dust_to_gas_ratio : float
        The dust-to-gas ratio.
    velocity_x_gas: float
        The initial uniform gas velocity in the x-direction.
    velocity_x_gas: float
        The initial uniform dust velocity in the x-direction.
    hfact: float
        The smoothing length factor 'hfact'.
    """

    number_of_dust_species = len(dust_to_gas_ratio)
    dust_types = tuple(range(idust, idust + number_of_dust_species))

    particle_types = (igas, *dust_types)

    npartx = {igas: number_of_particles_in_x_gas}
    npartx.update({idx: number_of_particles_in_x_dust for idx in dust_types})

    density_dust = [eps * density_gas for eps in dust_to_gas_ratio]
    rho = {igas: density_gas}
    rho.update(dict(zip(dust_types, density_dust)))

    velx = {igas: velocity_x_gas}
    velx.update({idx: velocity_x_dust for idx in dust_types})

    for itype in particle_types:
        _add_particle_of_type_to_box(
            setup, itype, npartx[itype], rho[itype], velx[itype], hfact
        )


def _add_particle_of_type_to_box(
    setup: phantomsetup.Setup,
    particle_type: int,
    npartx: int,
    rho: float,
    velx: float,
    hfact: float,
) -> None:
    """
    Helper function to add particles of a particular type to the box.

    Parameters
    ----------
    setup : phantomsetup.Setup
        The phantomsetup object representing the simulation.
    particle_type: int
        The particle type to add.
    npartx: int
        The number of particles in the x-direction. Thus the total
        number of particles for a cube box would be roughly npartx**3.
    rho: float
        The initial density.
    velx: float
        The initial uniform velocity in the x-direction.
    hfact: float
        The smoothing length factor 'hfact'.
    """

    # Particle positions
    particle_spacing = setup.box.xwidth / npartx
    position, smoothing_length = phantomsetup.distributions.uniform_distribution(
        boundary=setup.box.boundary, particle_spacing=particle_spacing, hfact=hfact
    )
    npart = position.shape[0]

    # Particle mass
    particle_mass = rho * setup.box.volume / npart

    # Particle velocities
    velocity = np.hstack((velx * np.ones((npart, 1)), np.zeros((npart, 2))))

    # Add particles
    setup.add_particles(
        particle_type, particle_mass, position, velocity, smoothing_length
    )


if __name__ == '__main__':
    dustybox = main()
