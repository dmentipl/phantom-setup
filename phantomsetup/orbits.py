

def hill_sphere_radius(
    planet_radius: float,
    planet_mass: float,
    stellar_mass: float,
    eccentricity: float = None,
) -> float:
    """
    Calculate the Hill sphere radius.

    Parameters
    ----------
    planet_radius
        The orbital radius of the planet.
    planet_mass
        The mass of the planet.
    stellar_mass
        The mass of the star.

    Optional Parameters
    -------------------
    eccentricity
        The orbital eccentricity.

    Returns
    -------
    hill_radius
        The Hill sphere radius.
    """

    if eccentricity is None:
        eccentricity = 0.0

    return (
        (1 - eccentricity)
        * planet_radius
        * (planet_mass / (3 * stellar_mass)) ** (1 / 3)
    )
