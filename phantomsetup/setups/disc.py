"""
Setup an accretion disc.
"""

from dataclasses import dataclass, field

import numpy as np

from .. import constants, units
from ..defaults import particle_type
from ..parameters import ParametersBase
from ..phantomsetup import Setup
from ..utils import keplerian_angular_velocity

# ------------------------------------------------------------------------------------ #
# Constants

# igas and idust are Phantom integer types for particles
igas = particle_type['igas']
idust = particle_type['idust']


# ------------------------------------------------------------------------------------ #
# Parameters


@dataclass
class Parameters(ParametersBase):
    """Disc setup parameters."""

    # prefix
    default = 'disc'
    description = (
        'The file name prefix, such that the dump file is prefix_00000.tmp.h5 and the '
        'in file is prefix.in.'
    )
    prefix: str = field(default=default, metadata={'description': description})

    # length_unit
    default = 'au'
    description = 'Distance unit as a string.'
    length_unit: str = field(default=default, metadata={'description': description})

    # mass_unit
    default = 'solarm'
    description = 'Mass unit as a string.'
    mass_unit: str = field(default=default, metadata={'description': description})

    # stellar_mass
    default = 1.0
    description = 'Stellar mass.'
    stellar_mass: float = field(default=default, metadata={'description': description})

    # stellar_accretion_radius
    default = 1.0
    description = 'Stellar accretion radius.'
    stellar_accretion_radius: float = field(
        default=default, metadata={'description': description}
    )

    # ieos
    default = 3
    description = (
        'The equation of state specified by index. 1: globally isothermal, '
        '2: adiabatic, 3: locally isothermal.'
    )
    ieos: int = field(default=default, metadata={'description': description})

    # aspect_ratio
    default = 0.05
    description = 'Aspect ratio at reference radius.'
    aspect_ratio: float = field(default=default, metadata={'description': description})

    # reference_radius
    default = 1.0
    description = 'Reference radius at which to set the aspect ratio.'
    reference_radius: float = field(
        default=default, metadata={'description': description}
    )

    # q_index_sound_speed
    default = 0.75
    description = 'The exponent q in the power law for sound speed: c_s ~ R^-q.'
    q_index_sound_speed: float = field(
        default=default, metadata={'description': description}
    )

    # use_dust
    default = False
    description = 'Add dust or not.'
    use_dust: bool = field(default=default, metadata={'description': description})

    # dust_method
    default = 'largegrains'
    description = 'Dust method to use: either "largegrains" or "smallgrains".'
    dust_method: str = field(default=default, metadata={'description': description})

    # grain_size
    default = ()
    description = 'Grain size of each dust species as a tuple.'
    grain_size: tuple = field(default=default, metadata={'description': description})

    # grain_density
    default = 3.0
    description = 'Intrinsic grain density of the dust.'
    grain_density: float = field(default=default, metadata={'description': description})


# ------------------------------------------------------------------------------------ #
# Setup function for discs


def setup(parameters: Parameters) -> Setup:
    """
    Setup discs.

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

    length = units.unit_string_to_cgs(parameters.length_unit)
    mass = units.unit_string_to_cgs(parameters.mass_unit)
    setup.set_units(length, mass, constants.year / (2 * np.pi))

    # -------------------------------------------------------------------------------- #
    # Set equation of state

    polyk = (
        parameters.aspect_ratio
        * keplerian_angular_velocity(
            parameters.reference_radius, parameters.stellar_mass
        )
        * parameters.reference_radius ** parameters.q_index_sound_speed
    ) ** 2
    setup.set_equation_of_state(ieos=parameters.ieos, polyk=polyk)

    # -------------------------------------------------------------------------------- #
    # Add the star
    setup.add_sink(
        mass=parameters.stellar_mass,
        accretion_radius=parameters.stellar_accretion_radius,
    )

    # -------------------------------------------------------------------------------- #
    # Set dust grain size distribution

    if parameters.use_dust:
        setup.set_dust(
            dust_method=parameters.dust_method,
            drag_method='Epstein/Stokes',
            grain_size=parameters.grain_size,
            grain_density=parameters.grain_density,
        )

    # -------------------------------------------------------------------------------- #
    # Add particles to disc

    setup.add_disc()

    # -------------------------------------------------------------------------------- #
    # Add extra quantities to particles

    alpha = np.zeros(setup.total_number_of_particles, dtype=np.single)
    setup.add_array_to_particles('alpha', alpha)

    # -------------------------------------------------------------------------------- #
    # Return the phantomsetup.Setup object

    return setup
