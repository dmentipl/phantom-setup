from __future__ import annotations

import copy
import datetime
from pathlib import Path
from typing import Any, Dict, Set, Union

import h5py
import numpy as np
import phantomconfig

from . import defaults
from .eos import EquationOfState, ieos_isothermal
from .infile import generate_infile

FILEIDENT_LEN = 100

IGAS = defaults.particle_type['igas']
IDUST = defaults.particle_type['idust']
IDUSTLAST = defaults.particle_type['idustlast']


class Setup:
    """
    The initial conditions for a Phantom simulation.

    TODO: add to description

    Examples
    --------
    TODO: add examples
    """

    def __init__(self) -> None:
        super().__init__()

        self._setup: str = None
        self._prefix: str = None

        self._header: Dict[str, Any] = defaults.header

        self._particle_mass: Dict[int, float] = {}
        self._particle_type: np.ndarray = None
        self._position: np.ndarray = None
        self._smoothing_length: np.ndarray = None
        self._velocity: np.ndarray = None

        self._extra_arrays: Dict[str, Any] = {}

        self._dust_fraction: np.ndarray = None
        self._use_dust_fraction = False
        self.number_of_small_dust_types: int = 0

        self._eos: EquationOfState = None
        self._units: Dict[str, float] = None
        self._box: Box = None

        self._fileident: str = None
        self._compile_options: Dict[str, Any] = copy.deepcopy(defaults.compile_options)
        self.run_options: phantomconfig.PhantomConfig = copy.deepcopy(
            defaults.run_options
        )

    @property
    def prefix(self) -> str:
        """Prefix for dump file and in file."""
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        self._prefix = prefix

    def add_array_to_particles(self, name: str, array: np.ndarray) -> Setup:
        """
        Add an array to existing particles.

        Parameters
        ----------
        name : str
            The name of the array.
        array : np.ndarray
            The array, such that the first index is the particle index.

        See Also
        --------
        add_particles : Add particles to the setup.
        set_dust_fraction: To set the dust fraction.

        Examples
        --------
        Adding an array 'alpha' of scalar quantities on the particles.
        >>> npart = setup.number_of_particles
        >>> alpha = np.random.rand(npart)
        >>> setup.add_array_to_particles('alpha', alpha)
        """
        self._extra_arrays[name] = array
        return self

    @property
    def position(self) -> np.ndarray:
        """Cartesian positions of particles."""
        return self._position

    @property
    def smoothing_length(self) -> np.ndarray:
        """Smoothing length of particles."""
        return self._smoothing_length

    @property
    def velocity(self) -> np.ndarray:
        """Cartesian velocities of particles."""
        return self._velocity

    @property
    def particle_type(self) -> np.ndarray:
        """
        Type of each particle.
        """
        return self._particle_type

    @property
    def particle_types(self) -> Set[int]:
        """
        Particle types.
        """
        return set(np.unique(self._particle_type, return_counts=True)[0])

    @property
    def number_of_particles(self) -> Dict[int, int]:
        """Number of particles of each type."""
        types, counts = np.unique(self._particle_type, return_counts=True)
        return dict(zip(types, counts))

    @property
    def total_number_of_particles(self) -> int:
        """Total number of particles of each type."""
        return sum(self.number_of_particles.values())

    @property
    def particle_mass(self) -> Dict[int, float]:
        """The particle mass per particle type."""
        return self._particle_mass

    def add_particles(
        self,
        particle_type: int,
        particle_mass: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        smoothing_length: np.ndarray,
    ) -> Setup:
        """
        Add particles to initial conditions.

        Parameters
        ----------
        particle_type : int
            The integer representing the particle type.
        particle_mass : float
            The particle mass.
        positions : (N, 3) np.ndarray
            The particle positions as N x 3 array, where the 2nd
            component is the Cartesian position, i.e. x, y, z.
        velocities : (N, 3) np.ndarray
            The particle velocities as N x 3 array, where the 2nd
            component is the Cartesian velocity, i.e. vx, vy, vz.
        smoothing_length : (N,) np.ndarray
            The particle smoothing length as N x 1 array.

        See Also
        --------
        add_array_to_particles : Add an array to existing particles.
        set_dust_fraction: To set the dust fraction.
        """

        if positions.ndim != 2:
            raise ValueError('positions wrong shape, must be (N, 3)')
        if velocities.ndim != 2:
            raise ValueError('velocities wrong shape, must be (N, 3)')
        if smoothing_length.ndim != 1:
            raise ValueError('smoothing_length wrong shape, must be (N,)')
        if positions.shape[1] != 3:
            raise ValueError('positions wrong shape, must be (N, 3)')
        if velocities.shape[1] != 3:
            raise ValueError('velocities wrong shape, must be (N, 3)')
        if (
            positions.shape != velocities.shape
            or positions.shape[0] != smoothing_length.size
        ):
            raise ValueError(
                'positions, velocities, and smoothing_length must have the same'
                ' number of particles'
            )

        if self._position is not None:
            self._position = np.append(self._position, positions, axis=0)
        else:
            self._position = positions

        if self._smoothing_length is not None:
            self._smoothing_length = np.append(
                self._smoothing_length, smoothing_length, axis=0
            )
        else:
            self._smoothing_length = smoothing_length

        if self._velocity is not None:
            self._velocity = np.append(self._velocity, velocities, axis=0)
        else:
            self._velocity = velocities

        if self._particle_type is not None:
            self._particle_type = np.append(
                self._particle_type,
                particle_type * np.ones(positions.shape[0], dtype=np.int),
                axis=0,
            )
        else:
            self._particle_type = particle_type * np.ones(
                positions.shape[0], dtype=np.int
            )

        self._particle_mass.update({particle_type: particle_mass})

        return self

    def set_dust_fraction(self, dustfrac: np.ndarray) -> Setup:
        """
        Set the dust fraction on existing particles.

        Parameters
        ----------
        dustfrac : (N, M) np.ndarray
            The M dust fractions per dust species on N particles, where
            M is the number of dust species.

        See Also
        --------
        add_particles : Add particles to the setup.
        add_array_to_particles : Add an array to existing particles.
        """

        if dustfrac.ndim > 2:
            raise ValueError('dustfrac has wrong shape')
        if dustfrac.shape[0] != self.number_of_particles[IGAS]:
            raise ValueError(
                'dustfrac must have shape (N, M) where N is number of particles'
            )

        self._dust_fraction = dustfrac
        self.number_of_small_dust_types = self._dust_fraction.shape[1]
        self._use_dust_fraction = True

        return self

    @property
    def dust_fraction(self) -> np.ndarray:
        return self._dust_fraction

    def set_equation_of_state(self, ieos: int, **kwargs) -> Setup:
        """
        Set the equation of state.

        Parameters
        ----------
        ieos : int
            The equation of state as represented by the following integers:
                1: 'isothermal'
                2: 'adiabatic/polytropic'
                3: 'locally isothermal disc'

        See Also
        --------
        phantomsetup.eos.EquationOfState
        """
        self._eos = EquationOfState(ieos, **kwargs)
        if ieos in ieos_isothermal:
            self.set_compile_option('ISOTHERMAL', True)
        else:
            self.set_compile_option('ISOTHERMAL', False)
        return self

    @property
    def eos(self) -> None:
        """The equation of state."""
        return self._eos

    @property
    def box(self) -> Box:
        """The boundary box."""
        return self._box

    def set_boundary(self, boundary: tuple) -> Setup:
        """
        Set the boundary Cartesian box.

        Parameters
        ----------
        boundary : tuple
            The boundary of the box like
            (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        xmin, xmax, ymin, ymax, zmin, zmax = boundary
        self._box = Box(xmin, xmax, ymin, ymax, zmin, zmax)
        return self

    @property
    def units(self) -> Dict[str, float]:
        return self._units

    @units.setter
    def units(
        self, length: float = None, mass: float = None, time: float = None
    ) -> None:

        if length is None:
            if self._units['length'] is not None:
                length = self._units['length']
            else:
                length = 1.0
        if mass is None:
            if self._units['mass'] is not None:
                mass = self._units['mass']
            else:
                mass = 1.0
        if time is None:
            if self._units['time'] is not None:
                time = self._units['time']
            else:
                time = 1.0

        self._units = {'length': length, 'mass': mass, 'time': time}

    @property
    def compile_options(self) -> None:
        """Phantom compile time options."""
        return self._compile_options

    def set_compile_option(self, key, value):
        if key in self._compile_options:
            self._compile_options[key] = value
        else:
            raise ValueError(f'key={key} does not exist')

    @property
    def number_of_large_dust_types(self) -> int:
        """Number of '2-fluid', i.e. large dust species."""
        return len(
            [
                itype
                for itype, npartoftype in self.number_of_particles.items()
                if IDUST <= itype <= IDUSTLAST and npartoftype > 0
            ]
        )

    @property
    def contains_large_dust(self) -> bool:
        return any(
            [
                npart
                for itype, npart in self.number_of_particles.items()
                if itype >= IDUST and itype <= IDUSTLAST and npart > 0
            ]
        )

    @property
    def fileident(self) -> None:
        """File information 'fileident' as defined in Phantom."""
        if self._fileident is None:
            self._generate_fileident()
        return self._fileident

    def _generate_fileident(self):

        fileident = (
            f'fulldump: Phantom '
            f'{defaults.PHANTOM_VERSION.split(".")[0]}.'
            f'{defaults.PHANTOM_VERSION.split(".")[1]}.'
            f'{defaults.PHANTOM_VERSION.split(".")[2]} '
            f'{defaults.PHANTOM_GIT_HASH} '
        )

        string = ''
        if self._compile_options['GRAVITY']:
            string += '+grav'
        if self.number_of_particles[IDUST] > 0:
            string += '+dust'
        if self._use_dust_fraction:
            string += '+1dust'
        if self._compile_options['H2CHEM']:
            string += '+H2chem'
        if self._compile_options['LIGHTCURVE']:
            string += '+lightcurve'
        if self._compile_options['DUSTGROWTH']:
            string += '+dustgrowth'

        if self._compile_options['MHD']:
            fileident += f'(mhd+clean{string}): '
        else:
            fileident += f'(hydro{string}): '

        fileident += datetime.datetime.strftime(
            datetime.datetime.today(), '%d/%m/%Y %H:%M:%S.%f'
        )[:-5]

        self._fileident = fileident

    def _update_header(self) -> None:
        """Update dump header for writing to file."""

        fileident = self.fileident.ljust(FILEIDENT_LEN).encode('ascii')
        self._header['fileident'] = fileident

        # Number and mass of particles

        self._header['nparttot'] = self.total_number_of_particles
        self._header['ntypes'] = defaults.maxtypes

        self._header['npartoftype'] = np.zeros(defaults.maxtypes, dtype=np.int)
        for key, val in self.number_of_particles.items():
            self._header['npartoftype'][key - 1] = val

        self._header['massoftype'] = np.zeros(defaults.maxtypes)
        for key, val in self.particle_mass.items():
            self._header['massoftype'][key - 1] = val

        # Dust

        self._header['ndustsmall'] = self.number_of_small_dust_types
        self._header['ndustlarge'] = self.number_of_large_dust_types

        # Equation of state

        if self._eos.polyk is not None:
            self._header['RK2'] = 3 / 2 * self._eos.polyk
        if self._eos.gamma is not None:
            self._header['gamma'] = self._eos.gamma
        if self._eos.qfacdisc is not None:
            self._header['qfacdisc'] = self._eos.qfacdisc

        if self._box is not None:
            self._header['xmin'] = self.box.xmin
            self._header['xmax'] = self.box.xmax
            self._header['ymin'] = self.box.ymin
            self._header['ymax'] = self.box.ymax
            self._header['zmin'] = self.box.zmin
            self._header['zmax'] = self.box.zmax

    def write_dump_file(self, filename: Union[str, Path] = None) -> Setup:
        """
        Write Phantom temporary ('.tmp') dump file.

        Optional parameters
        -------------------
        filename : str or Path
            The name of the dump file. Default is 'prefix_00000.tmp.h5'.
        """

        if filename is None:
            if self.prefix is None:
                raise ValueError('either choose a filename or set prefix value')
            else:
                filename = f'{self.prefix}_00000.tmp.h5'

        self._update_header()

        file_handle = h5py.File(filename, 'w')

        group = file_handle.create_group('header')
        for key, val in self._header.items():
            if isinstance(val, bytes):
                dset = group.create_dataset(key, (), dtype=f'S{FILEIDENT_LEN}')
                dset[()] = val
            else:
                group.create_dataset(name=key, data=val)

        group = file_handle.create_group('particles')

        group.create_dataset(name='xyz', data=self.position)
        group.create_dataset(name='h', data=self.smoothing_length)
        group.create_dataset(name='vxyz', data=self.velocity)
        group.create_dataset(name='itype', data=self.particle_type, dtype='i1')

        if self._dust_fraction is not None:
            group.create_dataset(name='dustfrac', data=self.dust_fraction)

        for name, array in self._extra_arrays.items():
            group.create_dataset(name=name, data=array)

        group = file_handle.create_group('sinks')

        file_handle.close()

        return self

    @property
    def infile(self) -> Dict[str, Any]:
        return generate_infile(self.compile_options, self.run_options, self._header)

    def write_in_file(self, filename: Union[str, Path] = None) -> Setup:
        """
        Write Phantom 'in' file.

        Optional parameters
        -------------------
        filename : str or Path
            The name of the in file. Default is 'prefix.in'.
        """

        if filename is None:
            filename = f'{self.prefix}.in'

        phantomconfig.read_dict(self.infile).write_phantom(filename)

        return self


class Box:
    """
    Cartesian box for simulations.

    Parameters
    ----------
    xmin : float
    xmax : float
    ymin : float
    ymax : float
    zmin : float
    zmax : float
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
    ) -> None:

        self.boundary = (xmin, xmax, ymin, ymax, zmin, zmax)

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

        self.xwidth = xmax - xmin
        self.ywidth = ymax - ymin
        self.zwidth = zmax - zmin

        self.volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
