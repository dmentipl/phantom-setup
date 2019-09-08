from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import h5py
import numpy as np
import phantomconfig as pc

from . import defaults

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

        self._infile = dict()
        self._header = defaults.header

        self._particle_type: np.ndarray = None
        self._position: np.ndarray = None
        self._smoothing_length: np.ndarray = None
        self._velocity: np.ndarray = None

        self._number_of_particles: List[int] = None
        self._particle_mass: Dict[int, float] = {}
        self._particle_types: Set[int] = None

        self._arrays: Dict[str, Any] = {}

        self._units: Dict[str, float] = None

    def add_array_to_particles(self, name: str, array: np.ndarray) -> None:
        """
        Add an array to existing particles.

        Parameters
        ----------
        name : str
            The name of the array.
        array : np.ndarray
            The array, such that the first index is the particle index.

        Examples
        --------
        Adding an array 'alpha' of scalar quantities on the particles.
        >>> npart = setup.number_of_particles
        >>> alpha = np.random.rand(npart)
        >>> setup.add_array_to_particles('alpha', alpha)
        """
        self._arrays[name] = array

    @property
    def position(self) -> None:
        """Cartesian positions of particles."""
        return self._position

    @property
    def smoothing_length(self) -> None:
        """Smoothing length of particles."""
        return self._smoothing_length

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
            types, counts = np.unique(self._particle_type, return_counts=True)
            return dict(zip(types, counts))
        return self._number_of_particles

    @property
    def total_number_of_particles(self) -> None:
        """Total number of particles of each type."""
        return sum(self.number_of_particles.values())

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
        self,
        itype: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        smoothing_length: np.ndarray,
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
        smoothing_length : (N,) np.ndarray
            The particle smoothing length as N x 1 array.
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

    def _update_header(self) -> None:
        """Update dump header for writing to file."""

        max_type = max(self.number_of_particles.keys())

        self._header['npartoftype'] = np.zeros(max_type, dtype=np.int)
        for key, val in self.number_of_particles.items():
            self._header['npartoftype'][key - 1] = val

        self._header['massoftype'] = np.zeros(max_type)
        for key, val in self.particle_mass.items():
            self._header['massoftype'][key - 1] = val

        self._header['nparttot'] = self.total_number_of_particles

    def write_dump_file(self, filename: Union[str, Path]) -> None:
        """
        Write Phantom temporary ('.tmp') dump file.

        Parameters
        ----------
        filename : str or Path
            The name of the dump file.
        """

        self._update_header()

        file_handle = h5py.File(filename, 'w')

        group = file_handle.create_group('header')
        for key, val in self._header.items():
            group.create_dataset(name=key, data=val)

        group = file_handle.create_group('particles')

        group.create_dataset(name='xyz', data=self.position)
        group.create_dataset(name='h', data=self.smoothing_length)
        group.create_dataset(name='vxyz', data=self.velocity)
        group.create_dataset(name='itype', data=self.particle_type, dtype='i1')

        for name, array in self._arrays.items():
            group.create_dataset(name=name, data=array)

        group = file_handle.create_group('sinks')

        file_handle.close()

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

        self._box: Tuple[float] = (1.0, 1.0, 1.0)

    @property
    def box(self) -> None:
        return self._box

    @box.setter
    def box(self, dx: float, dy: float, dz: float) -> None:
        self._box = (dx, dy, dz)
