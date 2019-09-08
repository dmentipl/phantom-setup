import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import phantomconfig as pc

_AVAILABLE_SETUPS = ('dustybox',)


class Setup:
    """
    The initial conditions for a Phantom simulation.

    Parameters
    ----------
    setup : str
        The problem to set up.
    """

    def __init__(self, setup: str) -> None:

        if setup not in _AVAILABLE_SETUPS:
            raise ValueError(f'Setup: {setup} not available')

        self.setup = setup
        self.infile = dict()
        self.header = dict.fromkeys(_HEADER_KEYS)

        self._particle_type: np.ndarray = None
        self._position: np.ndarray = None
        self._velocity: np.ndarray = None

        self._number_of_particles: List[int] = None
        self._particle_mass: Dict[int, float] = None
        self._particle_types: Set[int] = None

        self._units: Dict[str, float] = None

    @property
    def position(self) -> None:
        """Cartesian positions of particles."""
        return self._position

    @property
    def velocity(self) -> None:
        """Cartesian velocities of particles."""
        return self._velocity

    @property
    def particle_type(self) -> None:
        """
        Type of each particle.
        """
        return self._particle_type

    @property
    def particle_types(self) -> None:
        """
        Particle types.
        """
        if self._particle_types is None:
            return set(np.unique(self._particle_type, return_counts=True)[0])

    @property
    def number_of_particles(self) -> None:
        """Number of particles of each type."""
        if self._number_of_particles is None:
            return list(np.unique(self._particle_type, return_counts=True)[1])
        return self._number_of_particles

    @property
    def total_number_of_particles(self) -> None:
        """Total number of particles of each type."""
        return sum(self.number_of_particles)

    @property
    def particle_mass(self) -> None:
        """
        The particle mass per particle type.

        The data structure is Dict[int, float], where the key is the
        particle integer type, and the value is the mass as a float.
        """
        return self._particle_mass

    @particle_mass.setter
    def particle_mass(self, mass: Dict[int, float]) -> None:
        self._particle_mass = mass

    def add_particles(
        self, itype: int, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        """
        Add particles to initial conditions.

        Parameters
        ----------
        itype : int
            The integer representing the particle type.
        positions : (N, 3) np.ndarray
            The particle positions as N x 3 array, where the 2nd
            component is the Cartesian position, i.e. x, y, z.
        velocities : (N, 3) np.ndarray
            The particle velocities as N x 3 array, where the 2nd
            component is the Cartesian velocity, i.e. vx, vy, vz.
        """

        if positions.ndim != 2:
            raise ValueError('positions wrong shape, must be (N, 3)')
        if velocities.ndim != 2:
            raise ValueError('velocities wrong shape, must be (N, 3)')
        if positions.shape[1] != 3:
            raise ValueError('positions wrong shape, must be (N, 3)')
        if velocities.shape[1] != 3:
            raise ValueError('velocities wrong shape, must be (N, 3)')
        if positions.shape != velocities.shape:
            raise ValueError('positions and velocities must have same shape')

        if self._position is not None:
            self._position = np.append(self._position, positions, axis=0)
        else:
            self._position = positions

        if self._velocity is not None:
            self._velocity = np.append(self._velocity, velocities, axis=0)
        else:
            self._velocity = velocities

        if self._particle_type is not None:
            self._particle_type = np.append(
                self._particle_type,
                itype * np.ones(positions.shape[0], dtype=np.int),
                axis=0,
            )
        else:
            self._particle_type = itype * np.ones(positions.shape[0], dtype=np.int)

    @property
    def units(self) -> None:
        return self._units

    @units.setter
    def units(
        self, length: float = None, mass: float = None, time: float = None
    ) -> None:

        if length is None:
            length = self._units['length']
        if mass is None:
            mass = self._units['mass']
        if time is None:
            time = self._units['time']

        self._units = {'length': length, 'mass': mass, 'time': time}

    def write_temporary_dump_file(self, filename: Union[str, Path]) -> None:
        """
        Write Phantom temporary ('.tmp') dump file.
        """
        raise NotImplementedError

    def write_in_file(self, filename: Union[str, Path]) -> None:
        """
        Write Phantom 'in' file.

        Parameters
        ----------
        filename : str or pathlib.Path
            The name of the file to write to.
        """
        pc.read_dict(self.infile).write_phantom(filename)


class DustyBox(Setup):
    """
    Initial conditions for the DUSTYBOX test.
    """

    def __init__(self) -> None:
        super().__init__('dustybox')

        self._units: Dict[str, float] = {'length': 1.0, 'mass': 1.0, 'time': 1.0}
        self._box: Tuple[float] = (1.0, 1.0, 1.0)

    @property
    def box(self) -> None:
        return self._box

    @box.setter
    def box(self, dx: float, dy: float, dz: float) -> None:
        self._box = (dx, dy, dz)


_HEADER_KEYS = (
    'Bextx',
    'Bexty',
    'Bextz',
    'C_cour',
    'C_force',
    'alpha',
    'alphaB',
    'alphau',
    'angtot_in',
    'dtmax',
    'etot_in',
    'fileident',
    'gamma',
    'get_conserv',
    'graindens',
    'grainsize',
    'hfactfile',
    'ieos',
    'iexternalforce',
    'isink',
    'massoftype',
    'mdust_in',
    'ndustlarge',
    'ndustsmall',
    'npartoftype',
    'nparttot',
    'nptmass',
    'polyk',
    'polyk2',
    'qfacdisc',
    'rhozero',
    'tfile',
    'tolh',
    'totmom_in',
    'udist',
    'umass',
    'unit_Bfield',
    'utime',
    'xmax',
    'xmin',
    'ymax',
    'ymin',
    'zmax',
    'zmin',
)


_DEFAULT_INFILE = {
    # ------------------------------------------------
    # job name
    'logfile': 'prefix01.log',
    'dumpfile': 'prefix_00000.tmp',
    # ------------------------------------------------
    # options controlling run time and input/output
    'tmax': 0.1,
    'dtmax': 0.001,
    'nmax': -1,
    'nout': -1,
    'nmaxdumps': -1,
    'twallmax': datetime.timedelta(hours=0),
    'dtwallmax': datetime.timedelta(hours=12),
    'nfulldump': 1,
    'iverbose': 0,
    # ------------------------------------------------
    # options controlling accuracy
    'C_cour': 0.3,
    'C_force': 0.25,
    'tolv': 1.0e-2,
    'hfact': 1.0,
    'tolh': 1.0e-4,
    # ------------------------------------------------
    # options controlling hydrodynamics, artificial dissipation
    'alpha': 1.0,
    'alphamax': 1.0,
    'alphau': 1.0,
    'beta': 2.0,
    'alphaB': 1.0,
    'psidecayfac': 1.0,
    'overcleanfac': 1.0,
    'avdecayconst': 0.1,
    # ------------------------------------------------
    # options controlling damping
    'idamp': 0,
    'damp': 0.0,
    'tdyn_s': 0.0,
    # ------------------------------------------------
    # options controlling equation of state
    'ieos': 1,
    'mu': 2.381,
    'drhocrit': 0.50,
    'rhocrit0': 1.0e-18,
    'rhocrit1': 1.0e-14,
    'rhocrit2': 1.0e-10,
    'rhocrit3': 1.0e-3,
    'gamma1': 1.4,
    'gamma2': 1.1,
    'gamma3': 5 / 3,
    'rhocrit0pwp': 2.62780e12,
    'rhocrit1pwp': 5.01187e14,
    'rhocrit2pwp': 1.0e15,
    'gamma0pwp': 5 / 3,
    'gamma1pwp': 3.166,
    'gamma2pwp': 3.573,
    'gamma3pwp': 3.281,
    'p1pwp': 2.46604e34,
    'X': 0.74,
    'Z': 0.02,
    'relaxflag': 0,
    'ipdv_heating': 1,
    'ishock_heating': 1,
    'iresistive_heating': 1,
    # ------------------------------------------------
    # options controlling cooling
    'icooling': 0,
    'C_cool': 0.05,
    'beta_cool': 3.0,
    'dlq': 3.086e19,
    'dphot': 1.0801e20,
    'dphotflag': 0,
    'dchem': 3.086e20,
    'abundc': 1.4e-4,
    'abundo': 3.2e-4,
    'abundsi': 1.5e-5,
    'abunde': 2.0e-4,
    'uv_field_strength': 1.0,
    'dust_to_gas_ratio': 1.0,
    'AV_conversion_factor': 5.348e-22,
    'cosmic_ray_ion_rate': 1.0e-17,
    'iphoto': 1,
    'iflag_atom': 1,
    'cooltable': 'cooltable.dat',
    'habund': 0.7,
    'temp_floor': 1.0e4,
    # ------------------------------------------------
    # options controlling MCFOST
    'use_mcfost': False,
    'Voronoi_limits_file': 'limits',
    'use_mcfost_stars': False,
    # ------------------------------------------------
    # options controlling sink particles
    'icreate_sinks': 0,
    'rho_crit_cgs': 1.0e-10,
    'r_crit': 5.0e-3,
    'h_acc': 1.0e-3,
    'h_soft_sinkgas': 0.0,
    'h_soft_sinksink': 0.0,
    'f_acc': 0.8,
    # ------------------------------------------------
    # options relating to external forces
    # TODO: this is unfinished
    'iexternalforce': 0,
    # ------------------------------------------------
    # options controlling physical viscosity
    'irealvisc': 0,
    'shearparam': 0.1,
    'bulkvisc': 0.0,
    # ------------------------------------------------
    # options controlling forcing of turbulence
    # TODO
    # ------------------------------------------------
    # options controlling dust
    'idrag': 2,
    'K_code': 1.0,
    'grainsize': 1.0,
    'graindens': 1.0,
    'icut_backreaction': 0,
    'ilimitdustflux': False,
    # ------------------------------------------------
    # options controlling growth
    'ifrag': 0,
    'grainsizemin': 1.0e-3,
    'isnow': 0,
    'rsnow': 100.0,
    'Tsnow': 20.0,
    'vfrag': 15.0,
    'vfragin': 5.0,
    'vfragout': 15.0,
    # ------------------------------------------------
    # options controlling photoevaporation
    'mu_cgs': 1.26,
    'recombrate_cgs': 2.6e-13,
    'ionflux_cgs': 1.0e41,
    # ------------------------------------------------
    # options for injecting particles
    # TODO
    # ------------------------------------------------
    # options controlling non-ideal MHD
    # TODO
}
