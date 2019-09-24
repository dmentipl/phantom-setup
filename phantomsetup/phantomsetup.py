from __future__ import annotations

import copy
import datetime
import pathlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Collection, Dict, List, Tuple, Union

import h5py
import numpy as np
import phantomconfig
from phantomconfig import PhantomConfig

from . import constants, defaults
from .boundary import Boundary
from .box import Box
from .disc import Disc
from .eos import EquationOfState, ieos_isothermal
from .infile import generate_infile
from .particles import Particles
from .sinks import Sink

FILEIDENT_LEN = 100

IGAS = defaults.PARTICLE_TYPE['igas']


class Setup(Particles):
    """
    The initial conditions for a Phantom simulation.

    TODO: add to description

    Examples
    --------
    TODO: add examples
    """

    def __init__(self) -> None:
        super().__init__()

        self._prefix: str = None

        self._header: Dict[str, Any] = copy.deepcopy(defaults.HEADER)
        self._compile_options: Dict[str, Any] = copy.deepcopy(defaults.COMPILE_OPTIONS)
        self._run_options: phantomconfig.PhantomConfig = copy.deepcopy(
            defaults.RUN_OPTIONS
        )

        self._sinks: List[Sink] = None
        self._discs: List[Disc] = None
        self._boxes: List[Box] = None

        self._dust_method: str = None
        self._dust_fraction: np.ndarray = None
        self._grain_size: np.ndarray = np.array([])
        self._grain_density: np.ndarray = np.array([])
        self._number_of_small_dust_species: int = 0
        self._number_of_large_dust_species: int = 0

        self._eos: EquationOfState = None
        self._units: Dict[str, float] = None
        self._boundary = None

    @property
    def prefix(self) -> str:
        """Prefix for dump file and in file."""
        return self._prefix

    @prefix.setter
    def prefix(self, prefix: str) -> None:
        self._prefix = prefix
        self.set_run_option('logfile', f'{prefix}01.log')
        self.set_run_option('dumpfile', f'{prefix}_00000.tmp')

    @property
    def dust_fraction(self) -> np.ndarray:
        """The dust fraction for each dust species on the particles."""
        return self._dust_fraction

    @property
    def dust_method(self) -> str:
        """The dust method: either 'largegrains' or 'smallgrains'."""
        return self._dust_method

    @property
    def grain_size(self) -> np.array:
        """Grain sizes for each dust species."""
        return self._grain_size

    @property
    def grain_density(self) -> np.array:
        """Grain densities for each dust species."""
        return self._grain_density

    @property
    def number_of_small_dust_species(self) -> int:
        """Number of 'small' dust species."""
        return self._number_of_small_dust_species

    @number_of_small_dust_species.setter
    def number_of_small_dust_species(self, num) -> None:
        self._number_of_small_dust_species = num

    @property
    def number_of_large_dust_species(self) -> int:
        """Number of 'large' dust species."""
        return self._number_of_large_dust_species

    @number_of_large_dust_species.setter
    def number_of_large_dust_species(self, num) -> None:
        self._number_of_large_dust_species = num

    @property
    def sinks(self) -> List[Sink]:
        """Sink particles."""
        return self._sinks

    @property
    def number_of_sinks(self) -> int:
        if self._sinks is not None:
            return len(self._sinks)
        return 0

    @property
    def boxes(self) -> List[Box]:
        """Boxes of particles."""
        return self._boxes

    @property
    def number_of_boxes(self) -> int:
        if self._boxes is not None:
            return len(self._boxes)
        return 0

    @property
    def discs(self) -> List[Disc]:
        """Accretion discs."""
        return self._discs

    @property
    def number_of_discs(self) -> int:
        if self._discs is not None:
            return len(self._discs)
        return 0

    @property
    def eos(self) -> EquationOfState:
        """The equation of state."""
        return self._eos

    @property
    def boundary(self) -> Tuple[float, float, float, float, float, float]:
        """Simulation boundary (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return self._boundary

    @property
    def units(self) -> Dict[str, float]:
        return self._units

    @property
    def fileident(self) -> str:
        """File information 'fileident' as defined in Phantom."""
        return self._generate_fileident()

    @property
    def infile(self) -> Dict[str, Any]:
        return generate_infile(self.compile_options, self.run_options, self._header)

    @property
    def compile_options(self) -> Dict:
        """Phantom compile time options."""
        return self._compile_options

    @property
    def run_options(self) -> PhantomConfig:
        """Phantom run time options.

        This is a PhantomConfig object."""
        return self._run_options

    def set_compile_option(self, option: str, value: Any) -> None:
        """
        Set a Phantom compile time option.

        Parameters
        ----------
        option : str
            The compile time option to set.
        value : Any
            The value to set the option to.
        """
        if option in self._compile_options:
            self._compile_options[option] = value
        else:
            raise ValueError(f'Compile time option={option} does not exist')

    def get_compile_option(self, option: str) -> Any:
        """
        Get the value of a Phantom compile time option.

        Parameters
        ----------
        option
            The compile time option to get.

        Returns
        -------
        The value of the option.
        """
        if option in self._compile_options:
            return self._compile_options[option]
        else:
            raise ValueError(f'Compile time option={option} does not exist')

    def set_run_option(self, option: str, value: Any) -> None:
        """
        Set a Phantom run time option.

        Parameters
        ----------
        option : str
            The run time option to set.
        value : Any
            The value to set the option to.
        """
        if option in self._run_options.config:
            self._run_options.change_value(option, value)
        else:
            raise ValueError(f'Run time option={option} does not exist')

    def get_run_option(self, option: str) -> Any:
        """
        Get the value of a Phantom run time option.

        Parameters
        ----------
        option
            The run time option to get.

        Returns
        -------
        The value of the option.
        """
        if option in self._run_options.config:
            return self._run_options.config[option].value
        else:
            raise ValueError(f'Run time option={option} does not exist')

    def add_sink(
        self,
        *,
        mass: float,
        accretion_radius: float,
        position: tuple = None,
        velocity: tuple = None,
    ) -> Setup:
        """
        Add a sink particle.

        Parameters
        ----------
        mass : float
            The sink particle mass.
        accretion_radius : float
            The sink particle accretion radius.
        position : tuple
            The sink particle position.
        velocity : tuple
            The sink particle velocity.
        """

        if self._sinks is None:
            self._sinks = list()
        self._sinks.append(
            Sink(
                mass=mass,
                accretion_radius=accretion_radius,
                position=position,
                velocity=velocity,
            )
        )

        return self

    def add_box(self, box: Box) -> Setup:
        """
        Add a box of particles to the set up.

        Parameters
        ----------
        box
            The Box object.
        """

        if self._boxes is None:
            self._boxes = list()

        self.add_particles(
            particle_type=box.particle_type,
            particle_mass=box.particle_mass,
            positions=box.position,
            velocities=box.velocity,
            smoothing_length=box.smoothing_length,
        )

        self._boxes.append(box)

        return self

    def add_disc(self, disc) -> Setup:
        """
        Add a disc to the setup.

        Parameters
        ----------
        disc
            The Disc object.
        """

        if self._discs is None:
            self._discs = list()

        self.add_particles(
            particle_type=disc.particle_type,
            particle_mass=disc.particle_mass,
            positions=disc.position,
            velocities=disc.velocity,
            smoothing_length=disc.smoothing_length,
        )

        self._discs.append(disc)

        return self

    def set_dust(
        self,
        *,
        dust_method: str,
        drag_method: str,
        grain_size: Union[Collection, np.ndarray] = None,
        grain_density: float = None,
        drag_constant: float = None,
        number_of_dust_species: int = None,
        cut_back_reaction: bool = None,
    ) -> Setup:
        """
        Set the dust method, grain sizes, and intrinsic grain density.

        Parameters
        ----------
        dust_method : str
            The dust method, either: 'largegrains' or 'smallgrains'. In
            Phantom, 'largegrains' corresponds to the two-fluid or
            multi-fluid method, and 'smallgrains' corresponds to the
            one-fluid or dustfrac method.
        drag_method : str
            The drag method: 'K_const', 'ts_const', 'Epstein/Stokes'.
        grain_size : Union[Collection, np.ndarray]
            The grain sizes of each dust species.
        grain_density : float
            The intrinsic dust grain density.
        drag_constant : float
            The drag constant if constant drag is used.
        number_of_dust_species : int
            If constant drag, the number of dust species must be set.
        cut_back_reaction : bool
            Cut the drag on the gas phase from the dust.

        See Also
        --------
        set_dust_fraction : Set the dust fraction.
        """

        if dust_method not in ('largegrains', 'smallgrains'):
            raise ValueError('dust_method must be "largegrains" or "smallgrains"')
        if drag_method not in ('off', 'Epstein/Stokes', 'K_const', 'ts_const'):
            raise ValueError(
                'drag_method must be "off", "Epstein/Stokes", "K_const", "ts_const"'
            )
        if drag_method != 'Epstein/Stokes' and grain_size is not None:
            raise ValueError('No need to set grain_size if using constant drag')
        if drag_method != 'Epstein/Stokes' and number_of_dust_species is None:
            raise ValueError('Need to set number_of_dust_species for constant drag')

        self._dust_method = dust_method

        if drag_method == 'off':
            self.set_run_option('idrag', 0)
        elif drag_method == 'Epstein/Stokes':
            self.set_run_option('idrag', 1)
        elif drag_method == 'K_const':
            self.set_run_option('idrag', 2)
        elif drag_method == 'ts_const':
            self.set_run_option('idrag', 3)

        if drag_constant is not None:
            self.set_run_option('K_code', drag_constant)

        if cut_back_reaction:
            self.set_run_option('icut_backreaction', 1)

        if grain_size is not None:
            grain_size = np.array(grain_size)
            self._grain_size = grain_size

            if dust_method == 'largegrains':
                self.number_of_large_dust_species = grain_size.size
            elif dust_method == 'smallgrains':
                self.number_of_small_dust_species = grain_size.size

            if grain_density is None:
                grain_density = self.get_run_option('graindens')
            self._grain_density = grain_density * np.ones_like(grain_size)

        else:
            self._grain_size = np.zeros(number_of_dust_species)
            self._grain_density = np.zeros(number_of_dust_species)
            if dust_method == 'largegrains':
                self.number_of_large_dust_species = number_of_dust_species
            elif dust_method == 'smallgrains':
                self.number_of_small_dust_species = number_of_dust_species

        self.set_compile_option('DUST', True)
        if dust_method == 'smallgrains':
            self.set_compile_option('MAXDUSTSMALL', number_of_dust_species)
            self.set_compile_option('MAXDUSTLARGE', 0)
        elif dust_method == 'largegrains':
            self.set_compile_option('MAXDUSTLARGE', number_of_dust_species)
            self.set_compile_option('MAXDUSTSMALL', 0)

        self.set_kernel(kernel='quintic')
        self.set_compile_option('KERNEL', self.kernel)
        self.set_run_option('hfact', self.hfact)

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
        set_dust : Set the dust method, grain sizes, and grain density.
        """

        if dustfrac.ndim > 2:
            raise ValueError('dustfrac has wrong shape')
        if dustfrac.shape[0] != self.number_of_particles[IGAS]:
            raise ValueError(
                'dustfrac must have shape (N, M) where N is number of gas particles'
            )
        if dustfrac.shape[1] != self.number_of_small_dust_species:
            raise ValueError(
                'dustfrac shape does not match the number of small grains set by '
                'set_dust'
            )
        if self.dust_method != 'smallgrains':
            raise ValueError(
                'Attempting to set dust fraction without setting dust_method to '
                '"smallgrains"'
            )

        self._dust_fraction = dustfrac

        self.set_compile_option('DUST', True)

        return self

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
        phantomsetup.eos.EquationOfState : The equation of state class.
        """
        self._eos = EquationOfState(ieos, **kwargs)
        if ieos in ieos_isothermal:
            self.set_compile_option('ISOTHERMAL', True)
        else:
            self.set_compile_option('ISOTHERMAL', False)
        return self

    def set_boundary(self, boundary: tuple, periodic: bool = False) -> Setup:
        """
        Set the boundary as a Cartesian box.

        Parameters
        ----------
        boundary
            The boundary of the box like
            (xmin, xmax, ymin, ymax, zmin, zmax).

        Optional Parameters
        -------------------
        periodic
            Set to True for periodic boundary conditions.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = boundary
        self._boundary = Boundary(*boundary)
        if periodic:
            self.set_compile_option('PERIODIC', True)
        return self

    def set_units(
        self,
        length: float = None,
        mass: float = None,
        time: float = None,
        gravitational_constant_is_unity: bool = False,
    ) -> Setup:
        """
        Set code units for simulation.

        Parameters
        ----------
        length : float
            The length unit in cgs. Default is 1.0 cm.
        mass : float
            The mass unit in cgs. Default is 1.0 g.
        time : float
            The time unit in cgs. Default is 1.0 s.
        gravitational_constant_is_unity
            Only specify two units, and the third is set by the
            constraint that the gravitational constant is unity.
        """

        if gravitational_constant_is_unity:

            if length is not None and mass is not None and time is not None:
                raise ValueError(
                    'Cannot set length, mass, and time units together if requiring '
                    'gravitational constant to be 1.0.'
                )

            if mass is not None and length is not None:
                time = np.sqrt(length ** 3 / (constants.G * mass))
            elif length is not None and time is not None:
                mass = length ** 2 / (constants.G * time ** 2)
            elif mass is not None and time is not None:
                length = (time ** 2 * constants.G * mass) ** (1 / 3)
            elif time is not None:
                length = 1.0
                mass = length ** 2 / (constants.G * time ** 2)
            else:
                length = 1.0
                mass = 1.0
                time = np.sqrt(length ** 3 / (constants.G * mass))

        else:

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

        self._units = {
            'length': float(length),
            'mass': float(mass),
            'time': float(time),
        }

        return self

    def set_output(
        self,
        tmax: float = None,
        dtmax: float = None,
        ndumps: int = None,
        nfulldump: int = None,
    ) -> Setup:
        """
        Set the simulation run time and output frequency.

        Parameters
        ----------
        tmax
            The simulation run time.
        dtmax
            The time between dump files.
        ndumps
            Set 'dtmax' as 'tmax / ndumps'.
        nfulldump
            Write a full dump every 'nfulldump' dumps.
        """
        if dtmax is not None and ndumps is not None:
            raise ValueError('Cannot set dtmax and ndumps at the same time')
        if tmax is not None:
            self.set_run_option('tmax', tmax)
        if dtmax is not None:
            self.set_run_option('dtmax', dtmax)
        if ndumps is not None:
            dtmax = self.get_run_option('tmax') / ndumps
            self.set_run_option('dtmax', dtmax)
        if nfulldump is not None:
            self.set_run_option('nfulldump', nfulldump)
        return self

    def set_dissipation(
        self,
        *,
        alpha: float = None,
        alphamax: float = None,
        alphau: float = None,
        alphaB: float = None,
        beta: float = None,
        avdecayconst: float = None,
        disc_viscosity: bool = False,
    ) -> Setup:
        """
        Set the numerical dissipation parameters.

        Parameters
        ----------
        alpha
            Minimum artificial viscosity parameter.
        alphamax
            Maximum artificial viscosity parameter.
        alphau
            Artificial conductivity parameter.
        beta
            Beta viscosity.
        alphaB
            Artificial resistivity parameter.
        avdecayconst
            Decay time constant for viscosity switches.
        disc_viscosity
            Whether or not to use disc viscosity.
        """

        if alpha is not None:
            self.set_run_option('alpha', alpha)
        if alphamax is not None:
            self.set_run_option('alphamax', alphamax)
        if alphau is not None:
            self.set_run_option('alphau', alphau)
        if beta is not None:
            self.set_run_option('beta', beta)
        if alphaB is not None:
            self.set_run_option('alphaB', alphaB)
        if avdecayconst is not None:
            self.set_run_option('avdecayconst', avdecayconst)
        if disc_viscosity is not None:
            self.set_compile_option('DISC_VISCOSITY', disc_viscosity)

        return self

    def write_dump_file(self, directory: Union[str, Path] = None) -> Setup:
        """
        Write Phantom temporary ('.tmp') dump file.

        Optional parameters
        -------------------
        directory
            The path to a directory to write the file. Default is
            current working directory.
        """

        if directory is None:
            directory = pathlib.Path().cwd()
        directory = pathlib.Path(directory).expanduser().resolve()

        if not directory.exists():
            directory.mkdir(parents=True)

        if self.prefix is None:
            raise ValueError('No prefix set')
        else:
            filename = f'{self.prefix}_00000.tmp.h5'

        file_handle = h5py.File(directory / filename, 'w')

        self._write_header(file_handle)
        self._write_particle_arrays(file_handle)
        self._write_sink_arrays(file_handle)

        file_handle.close()

        return self

    def write_in_file(self, directory: Union[str, Path] = None) -> Setup:
        """
        Write Phantom 'in' file.

        Optional parameters
        -------------------
        directory
            The path to a directory to write the file. Default is
            current working directory.
        """

        if directory is None:
            directory = pathlib.Path().cwd()
        directory = pathlib.Path(directory).expanduser().resolve()

        if not directory.exists():
            directory.mkdir(parents=True)

        if self.prefix is None:
            raise ValueError('No prefix set')
        else:
            filename = f'{self.prefix}.in'

        phantomconfig.read_dict(self.infile).write_phantom(directory / filename)

        return self

    def phantom_compile_command(self, system: str = None, hdf5root: str = None) -> str:
        """
        Show the Phantom Makefile command for this setup.

        Optional Parameters
        -------------------
        system
            The Phantom SYSTEM Makefile variable. Default is 'gfortran'.
        hdf5root
            The root directory of your HDF5 library installation.
            Default is '/usr/local/opt/hdf5'.

        Returns
        -------
        str
            The Phantom compile command as a string.
        """

        return ' \\\n  '.join(
            self._generate_phantom_compile_command(system=system, hdf5root=hdf5root)
        )

    def compile_phantom(
        self,
        phantom_dir: Union[str, Path],
        system: str = None,
        hdf5root: str = None,
        working_dir: Union[str, Path] = None,
    ) -> subprocess.CompletedProcess:
        """
        Compile Phantom for this setup.

        Parameters
        ----------
        phantom_dir
            The path to the Phantom repository directory.

        Optional Parameters
        -------------------
        system
            The Phantom SYSTEM Makefile variable. Default is 'gfortran'.
        hdf5root
            The root directory of your HDF5 library installation.
            Default is '/usr/local/opt/hdf5'.
        working_dir
            The working directory for the setup. After a successful
            build, the Phantom binary will be copied here.

        Returns
        -------
        subprocess.CompletedProcess
            A CompletedProcess object without output from the command,
            and success/fail codes, etc.
        """

        phantom_dir = pathlib.Path(phantom_dir).expanduser().resolve()
        if not phantom_dir.exists():
            raise ValueError('phantom_dir does not exist')

        if working_dir is None:
            working_dir = pathlib.Path().cwd()
        working_dir = pathlib.Path(working_dir).expanduser().resolve()
        if not working_dir.exists():
            raise ValueError('working_dir does not exist')

        result = subprocess.run(
            self._generate_phantom_compile_command(system=system, hdf5root=hdf5root),
            cwd=phantom_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            print('Compilation failed.')
        else:
            print('Compilation successful.')
            shutil.copy(phantom_dir / 'bin/phantom', working_dir)
            shutil.copy(phantom_dir / 'bin/phantom_version', working_dir)

        return result

    def _write_header(self, file_handle: h5py.File):

        self._update_header()

        group = file_handle.create_group('header')
        for key, val in self._header.items():
            if isinstance(val, bytes):
                dset = group.create_dataset(key, (), dtype=f'S{FILEIDENT_LEN}')
                dset[()] = val
            else:
                group.create_dataset(name=key, data=val)

    def _write_particle_arrays(self, file_handle: h5py.File):

        group = file_handle.create_group('particles')

        group.create_dataset(name='xyz', data=self.position)
        group.create_dataset(name='h', data=self.smoothing_length)
        group.create_dataset(name='vxyz', data=self.velocity)
        group.create_dataset(name='itype', data=self.particle_type, dtype='i1')

        if self._dust_fraction is not None:
            group.create_dataset(name='dustfrac', data=self.dust_fraction)

        for name, array in self._extra_arrays.items():
            group.create_dataset(name=name, data=array)

    def _write_sink_arrays(self, file_handle: h5py.File):

        group = file_handle.create_group('sinks')

        if self.number_of_sinks > 0:

            n = self.number_of_sinks
            m = np.zeros((1, n))
            h = np.zeros((1, n))
            xyz = np.zeros((3, n))
            vxyz = np.zeros((3, n))

            for idx, sink in enumerate(self.sinks):
                m[:, idx] = sink.mass
                h[:, idx] = sink.accretion_radius
                xyz[:, idx] = sink.position
                vxyz[:, idx] = sink.velocity

            group.create_dataset(name='xyz', data=xyz)
            group.create_dataset(name='m', data=m)
            group.create_dataset(name='h', data=h)
            group.create_dataset(name='hsoft', data=np.zeros((1, n)))
            group.create_dataset(name='maccreted', data=np.zeros((1, n)))
            group.create_dataset(name='spinxyz', data=np.zeros((3, n)))
            group.create_dataset(name='tlast', data=np.zeros((1, n)))
            group.create_dataset(name='vxyz', data=vxyz)

    def _generate_fileident(self):

        fileident = (
            f'fulldump: Phantom '
            f'{defaults.PHANTOM_VERSION.split(".")[0]}.'
            f'{defaults.PHANTOM_VERSION.split(".")[1]}.'
            f'{defaults.PHANTOM_VERSION.split(".")[2]} '
            f'{defaults.PHANTOM_GIT_HASH} '
        )

        string = ''
        if self.get_compile_option('GRAVITY'):
            string += '+grav'
        if self._dust_method == 'largegrains':
            string += '+dust'
        if self._dust_method == 'smallgrains':
            string += '+1dust'
        if self.get_compile_option('H2CHEM'):
            string += '+H2chem'
        if self.get_compile_option('LIGHTCURVE'):
            string += '+lightcurve'
        if self.get_compile_option('DUSTGROWTH'):
            string += '+dustgrowth'

        if self.get_compile_option('MHD'):
            fileident += f'(mhd+clean{string}): '
        else:
            fileident += f'(hydro{string}): '

        fileident += datetime.datetime.strftime(
            datetime.datetime.today(), '%d/%m/%Y %H:%M:%S.%f'
        )[:-5]

        return fileident

    def _update_header(self) -> None:
        """Update dump header for writing to file."""

        fileident = self.fileident.ljust(FILEIDENT_LEN).encode('ascii')
        self._header['fileident'] = fileident

        # Number and mass of particles

        self._header['nparttot'] = self.total_number_of_particles
        self._header['ntypes'] = defaults.MAXTYPES

        self._header['npartoftype'] = np.zeros(defaults.MAXTYPES, dtype=np.int)
        for key, val in self.number_of_particles.items():
            self._header['npartoftype'][key - 1] = val

        self._header['massoftype'] = np.zeros(defaults.MAXTYPES)
        for key, val in self.particle_mass.items():
            self._header['massoftype'][key - 1] = val

        # dtmax
        self._header['dtmax'] = self.get_run_option('dtmax')

        # Artificial dissipation

        self._header['alpha'] = self.get_run_option('alpha')
        self._header['alphaB'] = self.get_run_option('alphaB')
        self._header['alphau'] = self.get_run_option('alphau')

        # hfact

        self._header['hfact'] = self.get_run_option('hfact')

        # Sink particles

        self._header['nptmass'] = self.number_of_sinks

        # Dust

        self._header['ndustsmall'] = self.number_of_small_dust_species
        self._header['ndustlarge'] = self.number_of_large_dust_species
        self._header['grainsize'] = self.grain_size
        self._header['graindens'] = self.grain_density

        # Equation of state

        if self._eos.polyk is not None:
            self._header['RK2'] = 3 / 2 * self._eos.polyk
        if self._eos.gamma is not None:
            self._header['gamma'] = self._eos.gamma
        if self._eos.qfacdisc is not None:
            self._header['qfacdisc'] = self._eos.qfacdisc

        # Boundary

        if self._boundary is not None:
            self._header['xmin'] = self.boundary.xmin
            self._header['xmax'] = self.boundary.xmax
            self._header['ymin'] = self.boundary.ymin
            self._header['ymax'] = self.boundary.ymax
            self._header['zmin'] = self.boundary.zmin
            self._header['zmax'] = self.boundary.zmax

        # Units

        if self._units is not None:
            self._header['udist'] = self._units['length']
            self._header['umass'] = self._units['mass']
            self._header['utime'] = self._units['time']

    def _generate_phantom_compile_command(
        self, system: str = None, hdf5root: str = None
    ) -> List[str]:
        """
        Generate the Phantom Makefile command for this setup.

        Optional Parameters
        -------------------
        system
            The Phantom SYSTEM Makefile variable. Default is 'gfortran'.
        hdf5root
            The root directory of your HDF5 library installation.
            Default is '/usr/local/opt/hdf5'.

        Returns
        -------
        list of str
            A list of strings when joined together produce the Makefile
            command to compile Phantom corresponding to this setup. E.g.
            >>> ' '.join(setup.generate_compile_command())
        """

        COMPILE_OPTIONS_IFDEF = ('LIGHTCURVE',)

        if system is None:
            system = 'gfortran'
        if hdf5root is None:
            hdf5root = '/usr/local/opt/hdf5'

        phantom_compile_command = [
            'make',
            'SETUP=empty',
            f'SYSTEM={system}',
            'HDF5=yes',
            f'HDF5ROOT={hdf5root}',
        ]

        for option, value in self.compile_options.items():
            if isinstance(value, bool):
                if value:
                    phantom_compile_command.append(f'{option}=yes')
                else:
                    if not option in COMPILE_OPTIONS_IFDEF:
                        phantom_compile_command.append(f'{option}=no')
                    else:
                        continue
            elif isinstance(value, (int, str)):
                phantom_compile_command.append(f'{option}={value}')
            else:
                raise ValueError('Cannot determine Phantom compile command')

        return phantom_compile_command

    def __repr__(self) -> str:
        return f"PhantomSetup('{self.prefix}')"
