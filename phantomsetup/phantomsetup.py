"""Phantomsetup.

Contains Setup class.
"""

from __future__ import annotations

import copy
import datetime
import pathlib
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import phantomconfig
from numpy import ndarray
from phantomconfig import PhantomConfig

from . import constants, defaults
from .boundary import Boundary
from .box import Box
from .disc import Disc
from .eos import EquationOfState, ieos_isothermal
from .infile import generate_infile
from .sinks import Sink

FILEIDENT_LEN = 100

IGAS = defaults.PARTICLE_TYPE['igas']

KERNELS = defaults.KERNELS
KERNEL_HFACT = defaults.KERNEL_HFACT

Container = Union[Box, Disc]


class Setup:
    """The initial conditions for a Phantom simulation.

    TODO: add to description

    Examples
    --------
    TODO: add examples
    """

    def __init__(self) -> None:

        self._prefix: str

        self._header: Dict[str, Any] = copy.deepcopy(defaults.HEADER)
        self._compile_options: Dict[str, Any] = copy.deepcopy(defaults.COMPILE_OPTIONS)
        self._run_options: phantomconfig.PhantomConfig = copy.deepcopy(
            defaults.RUN_OPTIONS
        )

        self._kernel: str = 'cubic'
        self._hfact: float = KERNEL_HFACT['cubic']

        self._particle_containers: List[Any] = list()

        self._sinks: List[Sink] = list()

        self._dust_method: str = 'none'
        self._dust_fraction: ndarray = None
        self._grain_size: ndarray = np.array([])
        self._grain_density: ndarray = np.array([])
        self._number_of_small_dust_species: int = 0
        self._number_of_large_dust_species: int = 0

        self._eos: EquationOfState
        self._units: Dict[str, float]
        self._boundary: Optional[Boundary] = None

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
    def number_of_particles(self) -> int:
        """Particle number."""
        return sum(n for n in self.number_of_particles_of_type.values())

    @property
    def number_of_particles_of_type(self) -> Dict[int, int]:
        """Particle number of each type."""
        d: Dict[int, int] = dict()
        for container in self._particle_containers:
            for key, value in container.number_of_particles_of_type.items():
                if key in d.keys():
                    d[key] += value
                else:
                    d[key] = value
        return d

    @property
    def mass_of_particle_type(self) -> Dict[int, float]:
        """Particle mass of each type."""
        d: Dict[int, float] = dict()
        for container in self._particle_containers:
            for key, value in container.mass_of_particle_type.items():
                if key in d.keys():
                    d[key] += value
                else:
                    d[key] = value
        return d

    @property
    def dust_fraction(self) -> ndarray:
        """Dust fraction for each dust species on the particles."""
        return self._dust_fraction

    @property
    def dust_method(self) -> str:
        """Dust method: either 'largegrains' or 'smallgrains'."""
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
        """Return number of 'small' dust species."""
        return self._number_of_small_dust_species

    @number_of_small_dust_species.setter
    def number_of_small_dust_species(self, num) -> None:
        self._number_of_small_dust_species = num

    @property
    def number_of_large_dust_species(self) -> int:
        """Return number of 'large' dust species."""
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
        """Return number of sinks."""
        return len(self._sinks)

    @property
    def boxes(self) -> List[Box]:
        """Boxes of particles."""
        return [
            container
            for container in self._particle_containers
            if isinstance(container, Box)
        ]

    @property
    def discs(self) -> List[Disc]:
        """Accretion discs."""
        return [
            container
            for container in self._particle_containers
            if isinstance(container, Disc)
        ]

    @property
    def eos(self) -> EquationOfState:
        """Equation of state."""
        return self._eos

    @property
    def boundary(self) -> Optional[Boundary]:
        """Boundary."""
        return self._boundary

    @property
    def units(self) -> Dict[str, float]:
        """Physical units."""
        return self._units

    @property
    def fileident(self) -> str:
        """File information 'fileident' as defined in Phantom."""
        return self._generate_fileident()

    @property
    def infile(self) -> Dict[str, Any]:
        """Phantom in file."""
        return generate_infile(self.compile_options, self.run_options, self._header)

    @property
    def compile_options(self) -> Dict:
        """Phantom compile time options."""
        return self._compile_options

    @property
    def run_options(self) -> PhantomConfig:
        """Phantom run time options.

        This is a PhantomConfig object.
        """
        return self._run_options

    def set_compile_option(self, option: str, value: Any) -> None:
        """Set a Phantom compile time option.

        Parameters
        ----------
        option
            The compile time option to set.
        value
            The value to set the option to.
        """
        if option in self._compile_options:
            self._compile_options[option] = value
        else:
            raise ValueError(f'Compile time option={option} does not exist')

    def get_compile_option(self, option: str) -> Any:
        """Get the value of a Phantom compile time option.

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
        """Set a Phantom run time option.

        Parameters
        ----------
        option
            The run time option to set.
        value
            The value to set the option to.
        """
        if option in self._run_options.config:
            self._run_options.change_value(option, value)
        else:
            raise ValueError(f'Run time option={option} does not exist')

    def get_run_option(self, option: str) -> Any:
        """Get the value of a Phantom run time option.

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

    @property
    def kernel(self) -> str:
        """SPH kernel."""
        return self._kernel

    @property
    def hfact(self) -> float:
        """Smoothing length factor for the SPH kernel."""
        return self._hfact

    def set_kernel(self, kernel: str, hfact: float = None) -> Setup:
        """Set the SPH kernel.

        Parameters
        ----------
        kernel
            The kernel as a string.
        hfact
            The kernel smoothing length factor.
        """
        if kernel not in KERNELS:
            raise ValueError(f'kernel={kernel} not available')

        if hfact is None:
            hfact = KERNEL_HFACT[kernel]

        self._kernel = kernel
        self._hfact = hfact

        return self

    def add_sink(
        self,
        *,
        mass: float,
        accretion_radius: float,
        position: Tuple[float, float, float] = None,
        velocity: Tuple[float, float, float] = None,
    ) -> Setup:
        """Add a sink particle.

        Parameters
        ----------
        mass
            The sink particle mass.
        accretion_radius
            The sink particle accretion radius.
        position
            The sink particle position.
        velocity
            The sink particle velocity.
        """
        self._sinks.append(
            Sink(
                mass=mass,
                accretion_radius=accretion_radius,
                position=position,
                velocity=velocity,
            )
        )

        return self

    def add_container(self, container: Container) -> Setup:
        """Add a container of particles to the set up.

        Parameters
        ----------
        container
            The container object. Can be a Box or Disc, etc.
        """
        self._particle_containers.append(container)
        return self

    def set_dust(
        self,
        *,
        dust_method: str,
        drag_method: str,
        grain_size: Optional[Union[list, tuple, ndarray]] = None,
        grain_density: Optional[float] = None,
        drag_constant: Optional[Union[float, list, tuple, ndarray]] = None,
        number_of_dust_species: Optional[int] = None,
        cut_back_reaction: Optional[bool] = None,
    ) -> Setup:
        """Set the dust method, grain size, and intrinsic grain density.

        Parameters
        ----------
        dust_method
            The dust method, either: 'largegrains' or 'smallgrains'. In
            Phantom, 'largegrains' corresponds to the two-fluid or
            multi-fluid method, and 'smallgrains' corresponds to the
            one-fluid or dustfrac method.
        drag_method
            The drag method: 'K_const', 'ts_const', 'Epstein/Stokes'.
        grain_size
            The grain sizes of each dust species.
        grain_density
            The intrinsic dust grain density.
        drag_constant
            The drag constant(s) if constant drag is used.
        number_of_dust_species
            If constant drag, the number of dust species must be set.
        cut_back_reaction
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

        if number_of_dust_species is not None:
            ndust = number_of_dust_species

        if drag_method == 'off':
            self.set_run_option('idrag', 0)
        elif drag_method == 'Epstein/Stokes':
            self.set_run_option('idrag', 1)
        elif drag_method == 'K_const':
            self.set_run_option('idrag', 2)
        elif drag_method == 'ts_const':
            self.set_run_option('idrag', 3)

        if drag_constant is not None:
            if isinstance(drag_constant, float):
                self.set_run_option('K_code', drag_constant)
            elif isinstance(drag_constant, (tuple, list, ndarray)):
                if len(drag_constant) == 1:
                    self.set_run_option('K_code', drag_constant[0])
                else:
                    self._run_options.remove_variable('K_code')
                    for idx, val in enumerate(drag_constant):
                        self._run_options.add_variable(
                            f'K_code{idx+1}',
                            val,
                            'drag constant when constant drag is used',
                            'options controlling dust',
                        )

        if cut_back_reaction:
            self.set_run_option('icut_backreaction', 1)

        if grain_size is not None:
            ndust = len(grain_size)
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
            self._grain_size = np.zeros(ndust)
            self._grain_density = np.zeros(ndust)
            if dust_method == 'largegrains':
                self.number_of_large_dust_species = ndust
            elif dust_method == 'smallgrains':
                self.number_of_small_dust_species = ndust

        self.set_compile_option('DUST', True)
        if dust_method == 'smallgrains':
            self.set_compile_option('MAXDUSTSMALL', ndust)
            self.set_compile_option('MAXDUSTLARGE', 0)
        elif dust_method == 'largegrains':
            self.set_compile_option('MAXDUSTLARGE', ndust)
            self.set_compile_option('MAXDUSTSMALL', 0)

        self.set_kernel(kernel='quintic')
        self.set_compile_option('KERNEL', self.kernel)
        self.set_run_option('hfact', self.hfact)

        return self

    def set_dust_fraction(self, dustfrac: ndarray) -> Setup:
        """Set the dust fraction on existing particles.

        Parameters
        ----------
        dustfrac
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
        if dustfrac.shape[0] != self.number_of_particles_of_type[IGAS]:
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
        """Set the equation of state.

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
        self.set_run_option('ieos', ieos)
        for param, value in self._eos.parameters.items():
            try:
                self.set_run_option(param, value)
            except ValueError:
                pass
        return self

    def set_boundary(self, boundary: tuple, periodic: bool = False) -> Setup:
        """Set the boundary as a Cartesian box.

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
        """Set code units for simulation.

        Parameters
        ----------
        length
            The length unit in cgs. Default is 1.0 cm.
        mass
            The mass unit in cgs. Default is 1.0 g.
        time
            The time unit in cgs. Default is 1.0 s.

        Optional Parameters
        -------------------
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
        """Set the simulation run time and output frequency.

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
        """Set the numerical dissipation parameters.

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
        """Write Phantom temporary ('.tmp') dump file.

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
        """Write Phantom 'in' file.

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

    def phantom_compile_command(
        self,
        system: str = None,
        hdf5root: str = None,
        extra_compiler_arguments: Union[List[str]] = None,
    ) -> str:
        """Show the Phantom Makefile command for this setup.

        Optional Parameters
        -------------------
        system
            The Phantom SYSTEM Makefile variable. Default is 'gfortran'.
        hdf5root
            The root directory of your HDF5 library installation.
            Default is '/usr/local/opt/hdf5'.
        extra_compiler_arguments
            Extra compiler arguments as a list of strings.

        Returns
        -------
        str
            The Phantom compile command as a string.
        """
        compile_command = self._generate_phantom_compile_command(
            system=system, hdf5root=hdf5root
        )
        if extra_compiler_arguments is not None:
            for arg in extra_compiler_arguments:
                compile_command.append(arg)
        return ' \\\n  '.join(compile_command)

    def compile_phantom(
        self,
        phantom_dir: Union[str, Path],
        system: str = None,
        hdf5root: str = None,
        working_dir: Union[str, Path] = None,
        extra_compiler_arguments: Union[List[str]] = None,
    ) -> subprocess.CompletedProcess:
        """Compile Phantom for this setup.

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
        extra_compiler_arguments
            Extra compiler arguments as a list of strings.

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

        compile_command = self._generate_phantom_compile_command(
            system=system, hdf5root=hdf5root
        )
        if extra_compiler_arguments is not None:
            for arg in extra_compiler_arguments:
                compile_command.append(arg)
        result = subprocess.run(
            compile_command,
            cwd=phantom_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            print('Compilation failed.')
            print(compile_command)
            print(result.stderr.decode('utf-8'))
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

    def _check_container_consistency(self):
        arrays = set(self._particle_containers[0].arrays.keys())
        for container in self._particle_containers:
            if set(container.arrays.keys()) != arrays:
                raise ValueError('Particle containers have inconsistent arrays')

    def _name_mapper(self):
        d = {
            'position': 'xyz',
            'velocity': 'vxyz',
            'smoothing_length': 'h',
            'particle_type': 'itype',
            'particle_mass': None,
        }
        return d

    def _write_particle_arrays(self, file_handle: h5py.File):

        group = file_handle.create_group('particles')

        self._check_container_consistency()

        containers = self._particle_containers
        array_names = containers[0].arrays.keys()
        for name in array_names:
            if containers[0].arrays[name].ndim == 1:
                data = np.hstack([container.arrays[name] for container in containers])
            elif containers[0].arrays[name].ndim == 2:
                data = np.vstack([container.arrays[name] for container in containers])
            if name in self._name_mapper():
                _name = self._name_mapper()[name]
                if _name is None:
                    continue
            else:
                _name = name
            group.create_dataset(name=_name, data=data)

        if self._dust_fraction is not None:
            group.create_dataset(name='dustfrac', data=self.dust_fraction)

    def _write_sink_arrays(self, file_handle: h5py.File):

        group = file_handle.create_group('sinks')

        if self.number_of_sinks > 0:

            n = self.number_of_sinks
            m = np.zeros(n)
            h = np.zeros(n)
            xyz = np.zeros((n, 3))
            vxyz = np.zeros((n, 3))

            for idx, sink in enumerate(self.sinks):
                m[idx] = sink.mass
                h[idx] = sink.accretion_radius
                xyz[idx, :] = sink.position
                vxyz[idx, :] = sink.velocity

            group.create_dataset(name='xyz', data=xyz)
            group.create_dataset(name='m', data=m)
            group.create_dataset(name='h', data=h)
            group.create_dataset(name='hsoft', data=np.zeros(n))
            group.create_dataset(name='maccreted', data=np.zeros(n))
            group.create_dataset(name='spinxyz', data=np.zeros((n, 3)))
            group.create_dataset(name='tlast', data=np.zeros(n))
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

        self._header['nparttot'] = sum(self.number_of_particles_of_type.values())
        self._header['ntypes'] = defaults.MAXTYPES

        self._header['npartoftype'] = np.zeros(defaults.MAXTYPES, dtype=np.int)
        for key, val in self.number_of_particles_of_type.items():
            self._header['npartoftype'][key - 1] = val

        self._header['massoftype'] = np.zeros(defaults.MAXTYPES)
        for key, val in self.mass_of_particle_type.items():
            self._header['massoftype'][key - 1] = val

        # dtmax
        self._header['dtmax'] = self.get_run_option('dtmax')

        # Numerical parameters

        self._header['tolh'] = self.get_run_option('tolh')
        self._header['C_cour'] = self.get_run_option('C_cour')
        self._header['C_force'] = self.get_run_option('C_force')

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

        self._header['ieos'] = self._eos.ieos
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
        """Generate the Phantom Makefile command for this setup.

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
                    if option not in COMPILE_OPTIONS_IFDEF:
                        phantom_compile_command.append(f'{option}=no')
                    else:
                        continue
            elif isinstance(value, (int, str)):
                phantom_compile_command.append(f'{option}={value}')
            else:
                raise ValueError('Cannot determine Phantom compile command')

        return phantom_compile_command

    def __repr__(self) -> str:
        """Repr."""
        return f"PhantomSetup('{self.prefix}')"
