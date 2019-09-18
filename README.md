Phantom setup
=============

> phantom-setup: generate initial conditions for [Phantom](https://bitbucket.org/danielprice/phantom) simulations

+ Docs: https://phantom-setup.readthedocs.io/
+ Repo: https://www.github.com/dmentipl/phantom-setup

[![Build Status](https://travis-ci.org/dmentipl/phantom-setup.svg?branch=master)](https://travis-ci.org/dmentipl/phantom-setup)
[![Coverage Status](https://coveralls.io/repos/github/dmentipl/phantom-setup/badge.svg?branch=master)](https://coveralls.io/github/dmentipl/phantom-setup?branch=master)
[![Documentation Status](https://readthedocs.org/projects/phantom-setup/badge/?version=latest)](https://phantom-setup.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/phantomsetup)](https://pypi.org/project/phantomsetup/)
[![Anaconda Version](https://img.shields.io/conda/v/dmentipl/phantom-setup.svg)](https://anaconda.org/dmentipl/phantom-setup)

Install
-------

Install via conda.

```
conda install phantomsetup --channel dmentipl
```

Install via pip.

```
pip install phantomsetup
```

Usage
-----

To generate initial conditions for a Phantom simulation via a Python script with phantom-setup:

1. Instantiate a `phantomsetup.Setup` object.
2. Add particles, set arrays, units, equation of state, sinks, dust, and other parameters.
3. Write a Phantom HDF5 dump file containing the particle arrays.
4. Write a Phantom in file containing the run time parameters.
5. Compile Phantom with the correct Makefile variables.

Then run Phantom.

Examples
--------

The `examples` folder contains examples that you can run as a Python script or Jupyter notebook.
