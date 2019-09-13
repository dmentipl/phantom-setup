from . import constants


def unit_string_to_cgs(string: str) -> float:
    """
    Convert a unit string to cgs.

    Parameters
    ----------
    string
        The string to convert.

    Returns
    -------
    float
        The value in cgs.
    """

    # distance
    if string.lower() == 'au':
        return constants.au

    # mass
    if string.lower() in ('solarm', 'msun'):
        return constants.solarm

    # time
    if string.lower() in ('year', 'years', 'yr', 'yrs'):
        return constants.year

    raise ValueError('Cannot convert unit')
