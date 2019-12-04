"""
Phantom setup
=============

phantom-setup is a Python package for setting up Phantom calculations
with Python and HDF5.

Daniel Mentiplay, 2019.
"""

from . import box, defaults, disc, eos, geometry, orbits, units
from .box import Box
from .disc import Disc
from .phantomsetup import Setup

__all__ = (
    'Box',
    'Disc',
    'Setup',
    'box',
    'defaults',
    'disc',
    'eos',
    'geometry',
    'orbits',
    'units',
)

# Canonical version number
__version__ = '0.0.1'
