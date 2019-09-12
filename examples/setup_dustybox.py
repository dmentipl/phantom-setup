"""
Setup the DUSTYBOX test problem.

The dust, possibly with multiple species, and gas are co-located in a
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

import argparse
import dataclasses
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Union

import numpy as np
import phantomsetup
import toml

# ------------------------------------------------------------------------------------ #
# Constants

# igas and idust are Phantom integer types for particles
igas = phantomsetup.defaults.particle_type['igas']
idust = phantomsetup.defaults.particle_type['idust']

# constant values are in cgs
au = phantomsetup.constants.au
solarm = phantomsetup.constants.solarm
year = phantomsetup.constants.year / (2 * np.pi)

# hfact=1.0 is for the quintic kernel which is best practice for
# calculations with dust
hfact = 1.0

# ieos=1 sets the isothermal equation of state
ieos = 1


# ------------------------------------------------------------------------------------ #
# Parameters


@dataclass
class Parameters:
    """DUSTYBOX setup parameters."""

    prefix: str = 'dustybox'

    box_boundary: tuple = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)

    sound_speed: float = 1.0

    number_of_particles_in_x_gas: int = 32
    number_of_particles_in_x_dust: int = 16

    density_gas: float = 1.0
    dust_to_gas_ratio: tuple = (0.01, 0.02, 0.03, 0.04, 0.05)

    drag_method: str = 'K_const'
    K_code: float = 1.0
    grain_size: tuple = ()
    grain_density: float = 0.0

    velocity_x_gas: float = 0.0
    velocity_x_dust: float = 1.0


# ------------------------------------------------------------------------------------ #
# Setup function for DUSTYBOX


def setup_dustybox(parameters: Parameters) -> phantomsetup.Setup:
    """
    Setup DUSTYBOX.

    This function sets up the DUSTYBOX test problem. It reads in
    parameters, instantiates the phantomsetup.Setup object, and adds
    particles and other information, then writes a Phantom initial
    condition dump and Phantom in file.

    Parameters
    ----------
    parameters : Parameters
        The parameters for this particular problem.

    Returns
    -------
    setup : phantomsetup.Setup
        The phantomsetup.Setup object.
    """

    # -------------------------------------------------------------------------------- #
    # Instantiate phantomsetup object

    setup = phantomsetup.Setup()

    # -------------------------------------------------------------------------------- #
    # File prefix

    setup.prefix = parameters.prefix

    # -------------------------------------------------------------------------------- #
    # Set unit

    setup.set_units(au, solarm, year)

    # -------------------------------------------------------------------------------- #
    # Set equation of state

    setup.set_equation_of_state(ieos=ieos, polyk=parameters.sound_speed ** 2)

    # -------------------------------------------------------------------------------- #
    # Set dust grain size distribution

    if parameters.drag_method == 'Epstein/Stokes':
        setup.set_dust(
            dust_method='largegrains',
            drag_method=parameters.drag_method,
            grain_size=parameters.grain_size,
            grain_density=parameters.grain_density,
        )

    elif parameters.drag_method in ('K_const', 'ts_const'):
        setup.set_dust(
            dust_method='largegrains',
            drag_method=parameters.drag_method,
            drag_constant=parameters.K_code,
            number_of_dust_species=len(parameters.dust_to_gas_ratio),
        )

    # -------------------------------------------------------------------------------- #
    # Setup box

    setup.set_boundary(parameters.box_boundary)

    # -------------------------------------------------------------------------------- #
    # Add dust and gas particles to box

    add_particles_to_box(
        setup,
        parameters.number_of_particles_in_x_gas,
        parameters.number_of_particles_in_x_dust,
        parameters.density_gas,
        parameters.dust_to_gas_ratio,
        parameters.velocity_x_gas,
        parameters.velocity_x_dust,
    )

    # -------------------------------------------------------------------------------- #
    # Add extra quantities to particles

    alpha = np.zeros(setup.total_number_of_particles, dtype=np.single)
    setup.add_array_to_particles('alpha', alpha)

    # -------------------------------------------------------------------------------- #
    # Write dump and in file

    setup.write_dump_file()
    setup.write_in_file()

    # -------------------------------------------------------------------------------- #
    # Return the phantomsetup.Setup object

    return setup


# ------------------------------------------------------------------------------------ #
# Helper functions


def add_particles_to_box(
    setup: phantomsetup.Setup,
    number_of_particles_in_x_gas: int,
    number_of_particles_in_x_dust: int,
    density_gas: float,
    dust_to_gas_ratio: Union[Collection, np.ndarray],
    velocity_x_gas: float,
    velocity_x_dust: float,
) -> None:
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
            setup, itype, npartx[itype], rho[itype], velx[itype]
        )

    return


def _add_particle_of_type_to_box(
    setup: phantomsetup.Setup, particle_type: int, npartx: int, rho: float, velx: float
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

    return


def read_parameters_from_file(filename: Union[str, Path]) -> Parameters:
    """
    Read parameters from TOML config file.

    Parameters
    ----------
    filename : str or Path
        The file name or path to the file to read.

    Returns
    -------
    Parameters
        The parameters from the file as a Parameters dataclass object.
    """

    filepath = pathlib.Path(filename).expanduser().resolve()
    t = toml.load(filepath)
    d = dict()
    for key, val in t.items():
        if isinstance(val, list):
            d[key] = tuple(val)
        else:
            d[key] = val
    return Parameters(**t)


def write_parameters_to_file(
    *, parameters: Parameters, filename: Union[str, Path]
) -> None:
    """
    Read parameters from TOML config file.

    Note: this does not preserve comments.

    Parameters
    ----------
    parameters : Parameters
        The parameters as a Parameters dataclass.
    filename : str or Path
        The file name or path to the file to read.
    """
    with open(filename, 'w') as fp:
        toml.dump(dataclasses.asdict(parameters), fp)
    return


# ------------------------------------------------------------------------------------ #
# Entry point for setup script

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Setup DUSTYBOX.')
    parser.add_argument('filename', type=Path, help='the setup parameter file')
    args = parser.parse_args()
    filename = args.filename

    if not filename.exists():
        print(f'No file named "{filename}" exists')
        if filename.suffix != '.toml':
            filename = filename.with_suffix('.toml')
        print(f'Writing default parameter file: {filename}')
        write_parameters_to_file(parameters=Parameters(), filename=filename)
    else:
        parameters = read_parameters_from_file(filename)
        setup = setup_dustybox(parameters)
