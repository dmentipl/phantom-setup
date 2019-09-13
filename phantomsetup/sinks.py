class Sink:
    """
    Sink particles.

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

    def __init__(
        self,
        *,
        mass: float,
        accretion_radius: float,
        position: tuple = None,
        velocity: tuple = None
    ):

        self._mass = mass
        self._accretion_radius = accretion_radius

        if position is not None:
            self._position = position
        else:
            self._position = (0.0, 0.0, 0.0)

        if velocity is not None:
            self._velocity = velocity
        else:
            self._velocity = (0.0, 0.0, 0.0)

    @property
    def mass(self) -> float:
        """Sink particle mass."""
        return self._mass

    @property
    def accretion_radius(self) -> float:
        """Sink particle accretion radius."""
        return self._accretion_radius

    @property
    def position(self) -> float:
        """Sink particle position."""
        return self._position

    @property
    def velocity(self) -> float:
        """Sink particle velocity."""
        return self._velocity
