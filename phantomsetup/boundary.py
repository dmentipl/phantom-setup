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
