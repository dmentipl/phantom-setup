"""
Setup the DUSTYBOX test problem.

The dust and gas are co-located in a box with uniform density. There is
an initial uniform differential velocity between the dust and gas.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Union

import numpy as np

from .. import constants
from ..defaults import particle_type
from ..distributions import uniform_distribution
from ..files import write_parameter_file
from ..phantomsetup import Setup

# ------------------------------------------------------------------------------------ #
# Constants

# igas and idust are Phantom integer types for particles
igas = particle_type['igas']
idust = particle_type['idust']

# constant values are in cgs
au = constants.au
solarm = constants.solarm
year = constants.year / (2 * np.pi)

# hfact should be 1.0 for the quintic kernel which is the appropriate
# kernel for calculations with dust
hfact = 1.0

# ieos of 1 sets the isothermal equation of state
ieos = 1


# ------------------------------------------------------------------------------------ #
# Parameters


@dataclass
class Parameters:
    """DUSTYBOX setup parameters."""

    # prefix
    default = 'dustybox'
    description = (
        'The file name prefix, such that the dump file is prefix_00000.tmp.h5 and the '
        'in file is prefix.in.'
    )
    prefix: str = field(default=default, metadata={'description': description})

    # box_boundary
    default = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    description = 'The boundary of the box as (xmin, xmax, ymin, ymax, zmin, zmax).'
    box_boundary: tuple = field(default=default, metadata={'description': description})

    # sound_speed
    default = 1.0
    description = 'The isothermal sound speed.'
    sound_speed: float = field(default=default, metadata={'description': description})

    # number_of_particles_in_x_gas
    default = 32
    description = (
        'The number of particles in the x direction. For a cube this is '
        'roughly the cubed root of the total number of particles.'
    )
    number_of_particles_in_x_gas: int = field(
        default=default, metadata={'description': description}
    )

    # number_of_particles_in_x_dust
    default = 16
    description = (
        'The number of particles in the x direction. For a cube this is roughly the '
        'cubed root of the total number of particles.'
    )
    number_of_particles_in_x_dust: int = field(
        default=default, metadata={'description': description}
    )

    # density_gas
    default = 1.0
    description = 'The initial uniform density of the gas.'
    density_gas: float = field(default=default, metadata={'description': description})

    # dust_to_gas_ratio
    default = (0.01, 0.02, 0.03, 0.04, 0.05)
    description = 'The dust-to-gas ratio for each dust species.'
    dust_to_gas_ratio: tuple = field(
        default=default, metadata={'description': description}
    )

    # drag_method
    default = 'K_const'
    description = (
        'The dust drag method. Options are "Epstein/Stokes", "K_const", or "ts_const".'
    )
    drag_method: str = field(default=default, metadata={'description': description})

    # K_code
    default = 1.0
    description = 'The constant drag coefficient.'
    K_code: float = field(default=default, metadata={'description': description})

    # grain_size
    default = ()
    description = 'The grain size of each dust species.'
    grain_size: tuple = field(default=default, metadata={'description': description})

    # grain_density
    default = 0.0
    description = 'The intrinsic grain density.'
    grain_density: float = field(default=default, metadata={'description': description})

    # velocity_x_gas
    default = 0.0
    description = 'The initial uniform velocity of gas.'
    velocity_x_gas: float = field(
        default=default, metadata={'description': description}
    )

    # velocity_x_dust
    default = 1.0
    description = 'The initial uniform velocity of dust.'
    velocity_x_dust: float = field(
        default=default, metadata={'description': description}
    )

    def check_consistency(self) -> None:
        """Check the parameters for consistency."""
        if self.drag_method == 'Epstein/Stokes':
            if len(self.grain_size) != len(self.dust_to_gas_ratio):
                raise ValueError(
                    'grain_size and dust_to_gas_ratio must have same length'
                )
        return

    def write_to_file(self, filename: Union[str, Path], *, overwrite: bool = False):
        """
        Write the parameters to TOML file.

        Parameters
        ----------
        filename : str or Path
            The name of the file to write. Should have extension
            '.toml'.
        overwrite : bool, default=False
            Whether to overwrite if the file exists.
        """
        write_parameter_file(
            self, filename, header='DUSTYBOX setup', overwrite=overwrite
        )


def get_parameters() -> Parameters:
    """Get the default parameters.

    Returns
    -------
    Parameters
        The default parameters as a Parameters dataclass object.
    """
    return Parameters()


# ------------------------------------------------------------------------------------ #
# Setup function for DUSTYBOX


def setup(parameters: Parameters) -> Setup:
    """
    Setup DUSTYBOX.

    This function sets up the DUSTYBOX test problem. It reads in
    parameters, instantiates the phantomsetup.Setup object, and adds
    particles and other information.

    Parameters
    ----------
    parameters
        The parameters for this particular problem.

    Returns
    -------
    setup
        The phantomsetup.Setup object.
    """

    # -------------------------------------------------------------------------------- #
    # Check parameters for consistency

    parameters.check_consistency()

    # -------------------------------------------------------------------------------- #
    # Instantiate phantomsetup object

    setup = Setup()

    # -------------------------------------------------------------------------------- #
    # Add parameters data class

    setup.parameters = parameters

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

    _add_particles_to_box(
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
    # Return the phantomsetup.Setup object

    return setup


# ------------------------------------------------------------------------------------ #
# Helper functions


def _add_particles_to_box(
    setup: Setup,
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
    setup: Setup, particle_type: int, npartx: int, rho: float, velx: float
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
    position, smoothing_length = uniform_distribution(
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
