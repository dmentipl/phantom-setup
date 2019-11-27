"""Particles."""
from __future__ import annotations

from typing import Dict, Set

import numpy as np
from numpy import ndarray


class Particles:
    """The particles.

    TODO: add to description

    Examples
    --------
    TODO: add examples
    """

    _required_arrays = [
        'particle_type',
        'position',
        'velocity',
        'smoothing_length',
    ]

    def __init__(self):
        self.arrays: Dict[str, ndarray] = {arr: None for arr in self._required_arrays}
        self._mass_of_particle_type: Dict[int, float] = dict()

    def __len__(self):
        """Total number of particles."""
        return len(self.arrays['particle_type'])

    @property
    def particle_types(self) -> Set[int]:
        """Particle integer types as a set."""
        return set(np.unique(self.arrays['particle_type'], return_counts=True)[0])

    @property
    def number_of_particles(self) -> int:
        """Particle number."""
        return sum(n for n in self.number_of_particles_of_type.values())

    @property
    def number_of_particles_of_type(self) -> Dict[int, int]:
        """Particle number of each type."""
        types, counts = np.unique(self.arrays['particle_type'], return_counts=True)
        return dict(zip(types, counts))

    @property
    def mass_of_particle_type(self) -> Dict[int, float]:
        """Particle mass of each type."""
        return self._mass_of_particle_type

    def add_particles(
        self,
        particle_type: int,
        particle_mass: float,
        position: ndarray,
        velocity: ndarray,
        smoothing_length: ndarray,
        **kwargs,
    ) -> Particles:
        """Add particles to initial conditions.

        Parameters
        ----------
        particle_type : int
            The integer representing the particle type.
        particle_mass : float
            The particle mass.
        position : (N, 3) ndarray
            The particle positions in Cartesian coordinates.
        velocity : (N, 3) ndarray
            The particle velocities in Cartesian coordinates.
        smoothing_length : (N,) ndarray
            The particle smoothing lengths.

        Optional Parameters
        -------------------
        kwargs
            Any additional arrays as keyword arguments.

        See Also
        --------
        set_array : Add an array to existing particles.
        """
        if position.ndim != 2:
            raise ValueError('position wrong shape, must be (N, 3)')
        if velocity.ndim != 2:
            raise ValueError('velocity wrong shape, must be (N, 3)')
        if smoothing_length.ndim != 1:
            raise ValueError('smoothing_length wrong shape, must be (N,)')
        if position.shape[1] != 3:
            raise ValueError('position wrong shape, must be (N, 3)')
        if velocity.shape[1] != 3:
            raise ValueError('velocity wrong shape, must be (N, 3)')
        number_of_particles = smoothing_length.size
        if position.shape[0] != number_of_particles:
            raise ValueError(
                'position and smoothing_length must have the same number of particles'
            )
        if velocity.shape[0] != number_of_particles:
            raise ValueError(
                'velocity and smoothing_length must have the same number of particles'
            )
        for name, array in kwargs.items():
            if array.shape[0] != number_of_particles:
                raise ValueError(
                    f'{name} must have the same length as number of particles'
                )

        self._mass_of_particle_type[particle_type] = particle_mass
        _particle_type = particle_type * np.ones(position.shape[0], dtype=np.int8)

        names = [
            'particle_type',
            'position',
            'velocity',
            'smoothing_length',
        ]
        arrays = [_particle_type, position, velocity, smoothing_length]

        for name, array in kwargs.items():
            names.append(name)
            arrays.append(array)

        for name, array in zip(names, arrays):
            if self.arrays[name] is not None:
                self.arrays[name] = np.append(self.arrays[name], array, axis=0)
            else:
                self.arrays[name] = array

        return self

    def set_array(self, name: str, array: ndarray) -> Particles:
        """Set an array on existing particles.

        Parameters
        ----------
        name : str
            The name of the array.
        array : ndarray
            The array, such that the first index is the particle index.

        See Also
        --------
        add_particles : Add particles to the setup.

        Examples
        --------
        Adding an array 'alpha' of scalar quantities on the particles.
        >>> npart = setup.number_of_particles
        >>> alpha = np.random.rand(npart)
        >>> setup.set_array('alpha', alpha)
        """
        if array.shape[0] != len(self):
            raise ValueError('Array shape incompatible with existing particles')
        self.arrays[name] = array
        return self

    def check_arrays(self) -> None:
        """Check arrays for consistency."""
        number_of_particles = len(self)
        for name, array in self.arrays.items():
            if array.shape[0] != number_of_particles:
                raise ValueError(
                    f'{name} does not have same length as number of particles'
                )
        return
