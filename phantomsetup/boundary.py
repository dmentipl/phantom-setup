from __future__ import annotations

from typing import Tuple

from . import defaults

_AVAILABLE_DISTRIBUTIONS = ('cubic', 'close packed')
_HFACT_DEFAULT = defaults.RUN_OPTIONS.config['hfact'].value


class Boundary:
    """
    Cartesian boundary box for simulations.

    Parameters
    ----------
    xmin
        Minimum x value.
    xmax
        Maximum x value.
    ymin
        Minimum y value.
    ymax
        Maximum y value.
    zmin
        Minimum z value.
    zmax
        Maximum x value.
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
        super().__init__()

        self._boundary = (xmin, xmax, ymin, ymax, zmin, zmax)

        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._zmin = zmin
        self._zmax = zmax

        self._xwidth = xmax - xmin
        self._ywidth = ymax - ymin
        self._zwidth = zmax - zmin

        self._volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

    @property
    def boundary(self) -> Tuple[float, float, float, float, float, float]:
        """Box boundary (xmin, xmax, ymin, ymax, zmin, zmax)."""
        return self._boundary

    @property
    def volume(self) -> float:
        """Box volume."""
        return self._volume

    @property
    def xmin(self) -> float:
        return self._xmin

    @property
    def xmax(self) -> float:
        return self._xmax

    @property
    def ymin(self) -> float:
        return self._ymin

    @property
    def ymax(self) -> float:
        return self._ymax

    @property
    def zmin(self) -> float:
        return self._zmin

    @property
    def zmax(self) -> float:
        return self._zmax

    @property
    def xwidth(self) -> float:
        """Box width in x-direction."""
        return self._xwidth

    @property
    def ywidth(self) -> float:
        """Box width in y-direction."""
        return self._ywidth

    @property
    def zwidth(self) -> float:
        """Box width in z-direction."""
        return self._zwidth
