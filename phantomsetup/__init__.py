"""
Phantom setup
=============

phantom-setup is a Python package for setting up Phantom calculations
with Python and HDF5.

Daniel Mentiplay, 2019.
"""

from . import distributions, setups
from .phantomsetup import Setup

__all__ = ('Setup', 'distributions', 'setups')

# Canonical version number
__version__ = '0.0.1'
