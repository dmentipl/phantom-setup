"""
Setup the DUSTYBOX test problem.

The dust and gas are co-located in a box with uniform density. There is
an initial uniform differential velocity between the dust and gas.
"""

from dataclasses import dataclass, field

import numpy as np

from .. import constants
from ..box import Box
from ..defaults import particle_type
from ..parameters import ParametersBase
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
class Parameters(ParametersBase):
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

    # number_of_particles_gas
    default = 50_000
    description = 'The number of gas particles.'
    number_of_particles_gas: int = field(
        default=default, metadata={'description': description}
    )

    # number_of_particles_dust
    default = 10_000
    description = 'The number of dust particles in each species.'
    number_of_particles_dust: int = field(
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
    default = 3.0
    description = 'The intrinsic grain density.'
    grain_density: float = field(default=default, metadata={'description': description})

    # velocity_delta
    default = 1.0
    description = 'The initial delta in uniform velocity between gas and dust.'
    velocity_delta: float = field(
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
    # File prefix

    setup.prefix = parameters.prefix

    # -------------------------------------------------------------------------------- #
    # Set units

    setup.set_units(au, solarm, year)

    # -------------------------------------------------------------------------------- #
    # Set equation of state

    setup.set_equation_of_state(ieos=ieos, polyk=parameters.sound_speed ** 2)

    # -------------------------------------------------------------------------------- #
    # Set dust

    number_of_dust_species = len(parameters.dust_to_gas_ratio)

    density_dust = [
        eps * parameters.density_gas for eps in parameters.dust_to_gas_ratio
    ]

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
            number_of_dust_species=number_of_dust_species,
        )

    # -------------------------------------------------------------------------------- #
    # Set boundary

    setup.set_boundary(parameters.box_boundary)

    # -------------------------------------------------------------------------------- #
    # Add gas particles to box

    def velocity_distribution(xyz: np.ndarray) -> np.ndarray:
        """Gas has zero initial velocity."""
        vxyz = np.zeros_like(xyz)
        return vxyz

    box = Box(*parameters.box_boundary)
    box.add_particles(
        particle_type=igas,
        number_of_particles=parameters.number_of_particles_gas,
        density=parameters.density_gas,
        velocity_distribution=velocity_distribution,
        hfact=hfact,
    )
    setup.add_box(box)

    # -------------------------------------------------------------------------------- #
    # Add dust particles to box

    def velocity_distribution(xyz: np.ndarray) -> np.ndarray:
        """Dust has uniform initial velocity."""
        vxyz = np.zeros_like(xyz)
        vxyz[:, 0] = parameters.velocity_delta
        return vxyz

    for idx in range(number_of_dust_species):
        box = Box(*parameters.box_boundary)
        box.add_particles(
            particle_type=idust + idx,
            number_of_particles=parameters.number_of_particles_dust,
            density=density_dust[idx],
            velocity_distribution=velocity_distribution,
            hfact=hfact,
        )
        setup.add_box(box)

    # -------------------------------------------------------------------------------- #
    # Add extra quantities to particles

    alpha = np.zeros(setup.total_number_of_particles, dtype=np.single)
    setup.add_array_to_particles('alpha', alpha)

    # -------------------------------------------------------------------------------- #
    # Return the phantomsetup.Setup object

    return setup
