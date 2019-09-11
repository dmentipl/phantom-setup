"""
Phantom setup
=============

phantom-setup is a Python package for setting up Phantom calculations
with Python and HDF5.
"""

from . import constants
from . import particle_distributions as distributions
from .phantomsetup import Setup

__all__ = ('Setup', 'constants', 'distributions')

# Canonical version number
__version__ = '0.0.1'
