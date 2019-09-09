"""
Setup the DUSTYBOX test problem with multiple dust species.

The dust and gas are co-located in a box. The gas is initially
stationary, and the dust is given a uniform velocity to the right.
"""

import numpy as np
import phantomsetup


def main():

    # -------------------------------------------------------------------------------- #
    # Constants
    igas = phantomsetup.defaults.igas
    idust = phantomsetup.defaults.idust

    # -------------------------------------------------------------------------------- #
    # Parameters
    prefix = 'dustybox'
    box_boundary = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

    hfact = 1.0

    number_of_particles_in_x_gas = 32
    number_of_particles_in_x_dust = 16

    density_gas = 1.0
    density_dust = (0.01, 0.02, 0.03, 0.04, 0.05)

    velocity_x_gas = 0.0
    velocity_x_dust = 1.0

    # -------------------------------------------------------------------------------- #
    # Instantiate phantomsetup object
    dustybox = phantomsetup.Setup()
    dustybox.prefix = prefix

    # -------------------------------------------------------------------------------- #
    # Setup box
    dustybox.box = box_boundary

    # -------------------------------------------------------------------------------- #
    # Setup particles

    number_of_dust_species = len(density_dust)
    dust_types = tuple(range(idust, idust + number_of_dust_species))

    particle_types = (igas, *dust_types)

    npartx = {igas: number_of_particles_in_x_gas}
    npartx.update({idx: number_of_particles_in_x_dust for idx in dust_types})

    rho = {igas: density_gas}
    rho.update(dict(zip(dust_types, density_dust)))

    velx = {igas: velocity_x_gas}
    velx.update({idx: velocity_x_dust for idx in dust_types})

    for itype in particle_types:
        add_particle_of_type_to_box(
            dustybox, itype, npartx[itype], rho[itype], velx[itype], hfact
        )

    # -------------------------------------------------------------------------------- #
    # TODO: add other arrays...
    alpha = np.zeros(dustybox.total_number_of_particles)
    dustybox.add_array_to_particles('alpha', alpha)

    # -------------------------------------------------------------------------------- #
    # TODO: add other header items...

    # -------------------------------------------------------------------------------- #
    # Write dump and in file
    dustybox.write_dump_file()
    dustybox.write_in_file()


def add_particle_of_type_to_box(
    setup: phantomsetup.Setup,
    particle_type: int,
    npartx: int,
    rho: float,
    velx: float,
    hfact: float,
) -> None:
    """
    Add particles of a particular type to the box.

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
    position, smoothing_length = phantomsetup.dist.uniform_distribution(
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
    main()
