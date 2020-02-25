=================
API documentation
=================

.. currentmodule:: phantomsetup

phantomsetup
------------

.. autoclass:: phantomsetup.Setup


box
---

.. autoclass:: phantomsetup.box.Box
.. autofunction:: phantomsetup.box.uniform_distribution

eos
---

.. autoclass:: phantomsetup.eos.EquationOfState
.. autofunction:: phantomsetup.eos.polyk_for_locally_isothermal_disc

geometry
--------

.. autofunction:: phantomsetup.geometry.stretch_map
.. autofunction:: phantomsetup.geometry.coordinate_transform

infile
------

.. autofunction:: phantomsetup.infile.generate_infile

orbits
------

.. autofunction:: phantomsetup.orbits.hill_sphere_radius
.. autofunction:: phantomsetup.orbits.binary_orbit
.. autofunction:: phantomsetup.orbits.flyby_orbit
.. autofunction:: phantomsetup.orbits.flyby_time

particles
---------

.. autoclass:: phantomsetup.particles.Particles

sinks
-----

.. autoclass:: phantomsetup.sinks.Sink

units
-----

.. autofunction:: phantomsetup.units.unit_string_to_cgs
