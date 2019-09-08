"""
Phantom setup
=============

phantom-setup is a Python package for setting up Phantom calculations
with Python and HDF5.
"""

from . import particle_distributions as dist

from .phantomsetup import Setup

__all__ = ('Setup', 'dist')

# Canonical version number
__version__ = '0.0.1'
