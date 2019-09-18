from __future__ import annotations

from typing import Any, Dict, Set

import numpy as np

from . import defaults

IGAS = defaults.PARTICLE_TYPE['igas']
IDUST = defaults.PARTICLE_TYPE['idust']
IDUSTLAST = defaults.PARTICLE_TYPE['idustlast']


class Particles:
    """
    The particles.

    TODO: add to description

    Examples
    --------
    TODO: add examples
    """

    def __init__(self):

        self._particle_type: np.ndarray = None
        self._particle_mass: Dict[int, float] = {}
        self._position: np.ndarray = None
        self._smoothing_length: np.ndarray = None
        self._velocity: np.ndarray = None

        self._extra_arrays: Dict[str, Any] = {}

    @property
    def particle_type(self) -> np.ndarray:
        """Integer type of each particle."""
        return self._particle_type

    @property
    def particle_types(self) -> Set[int]:
        """Particle integer types as a set."""
        return set(np.unique(self._particle_type, return_counts=True)[0])

    @property
    def particle_mass(self) -> Dict[int, float]:
        """The particle mass per particle type."""
        return self._particle_mass

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
    def x(self) -> np.ndarray:
        """Cartesian x position of particles."""
        return self._position[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Cartesian y position of particles."""
        return self._position[:, 1]

    @property
    def z(self) -> np.ndarray:
        """Cartesian z position of particles."""
        return self._position[:, 2]

    @property
    def R(self) -> np.ndarray:
        """Cylindrical R position of particles."""
        return np.sqrt(self._position[:, 0] ** 2 + self._position[:, 1] ** 2)

    @property
    def phi(self) -> np.ndarray:
        """Cylindrical phi position of particles."""
        return np.arctan2(self._position[:, 1], self._position[:, 0])

    @property
    def h(self) -> np.ndarray:
        """Smoothing length of particles."""
        return self._smoothing_length

    @property
    def vx(self) -> np.ndarray:
        """Cartesian x velocity of particles."""
        return self._velocity[:, 0]

    @property
    def vy(self) -> np.ndarray:
        """Cartesian y velocity of particles."""
        return self._velocity[:, 1]

    @property
    def vz(self) -> np.ndarray:
        """Cartesian z velocity of particles."""
        return self._velocity[:, 2]

    @property
    def vR(self) -> np.ndarray:
        """Cylindrical R velocity of particles."""
        return (
            self._position[:, 0] * self._velocity[:, 0]
            + self._position[:, 1] * self._velocity[:, 1]
        ) / np.sqrt(self._position[:, 0] ** 2 + self._position[:, 1] ** 2)

    @property
    def vphi(self) -> np.ndarray:
        """Cylindrical phi position of particles."""
        return (
            self._position[:, 0] * self._velocity[:, 1]
            - self._position[:, 1] * self._velocity[:, 0]
        ) / np.sqrt(self._position[:, 0] ** 2 + self._position[:, 1] ** 2)

    def add_particles(
        self,
        particle_type: int,
        particle_mass: float,
        positions: np.ndarray,
        velocities: np.ndarray,
        smoothing_length: np.ndarray,
    ) -> Particles:
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
        # if particle_type in range(IDUST, IDUSTLAST + 1):
        #     if self.dust_method != 'largegrains':
        #         raise ValueError(
        #             'Adding "largegrains" dust without calling set_dust first'
        #         )
        #     if particle_type - IDUST + 1 > self.number_of_large_dust_species:
        #         raise ValueError(
        #             'particle_type is greater than what is available from the call to '
        #             'set_dust'
        #         )

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

    def add_array_to_particles(self, name: str, array: np.ndarray) -> Particles:
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

        Examples
        --------
        Adding an array 'alpha' of scalar quantities on the particles.
        >>> npart = setup.number_of_particles
        >>> alpha = np.random.rand(npart)
        >>> setup.add_array_to_particles('alpha', alpha)
        """
        self._extra_arrays[name] = array
        return self
