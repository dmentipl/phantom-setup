---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Set up a disc

In this tutorial we set up a protoplanetary disc around a star represented by a sink particle, and we add a planet. This notebook generates a Phantom "temporary" dump file that can be read by Phantom as an initial condition. It also generates a Phantom "in" file. Together, these files can start a Phantom simulation.

## Initialization

First we import the required modules.

```python
import matplotlib.pyplot as plt
import numpy as np
import phantomsetup
from phantomsetup import defaults
from phantomsetup.disc import add_gap, self_similar_accretion_disc
from phantomsetup.eos import polyk_for_locally_isothermal_disc
from phantomsetup.orbits import hill_sphere_radius
from phantomsetup.units import unit_string_to_cgs
```

Here we set some constants for convenience.

```python
igas = defaults.particle_type['igas']
```

## Parameters

Now we set the parameters for the problem.

First is the `prefix` which sets the file name for the dump file and Phantom in file.

```python
prefix = 'disc'
```

### Resolution

We choose the resolution to be $10^6$ gas particles.

```python
number_of_particles = 1_000_000
particle_type = igas
```

### Viscosity

The SPH $\alpha$ viscosity parameter is its minimal value of 0.1.

```python
alpha_artificial = 0.1
```

### Units

We set the length and mass units to be au and solar masses, respectively. We will also set the time unit such that the gravitational constant is unity.

```python
length_unit = unit_string_to_cgs('au')
mass_unit = unit_string_to_cgs('solarm')
gravitational_constant = 1.0
```

### Star

The star is of solar mass, at the origin, with a 5 au accretion radius.

```python
stellar_mass = 1.0
stellar_accretion_radius = 5.0
stellar_position = (0.0, 0.0, 0.0)
stellar_velocity = (0.0, 0.0, 0.0)
```

### Disc

The disc has mass 0.01 solar masses, it extends from 10 au to 200 au.

```python
radius_min = 10.0
radius_max = 200.0

disc_mass = 0.01
```

### Equation of state

The equation of state is locally isothermal. We set the aspect ratio H/R at a reference radius.

```python
ieos = 3
q_index = 0.75
aspect_ratio = 0.05
reference_radius = 10.0
```

### Planet

We add a planet at 100 au.

```python
planet_mass = 0.001
planet_position = (100.0, 0.0, 0.0)
```

We set the planet accretion radius as a fraction of the Hill sphere radius.

```python
planet_accretion_radius_fraction_hill_radius = 0.25

orbital_radius = np.linalg.norm(planet_position)
planet_hill_radius = hill_sphere_radius(orbital_radius, planet_mass, stellar_mass)
planet_accretion_radius = (
    planet_accretion_radius_fraction_hill_radius * planet_hill_radius
)

planet_velocity = np.sqrt(gravitational_constant * stellar_mass / orbital_radius)
```

### Surface density distribution

For the surface density distribution we use the Lynden-Bell and Pringle (1974) self-similar solution, i.e. a power law with an exponential taper. Plus we add a gap, as a step function at the planet location.

```python
gap_width = planet_hill_radius

@add_gap(orbital_radius=orbital_radius, gap_width=gap_width)
def density_distribution(radius, radius_critical, gamma):
    """Surface density distribution.

    Self-similar disc solution with a gap added.
    """
    return self_similar_accretion_disc(radius, radius_critical, gamma)


radius_critical = 100.0
gamma = 1.0

args = (radius_critical, gamma)
```

## Instantiate the `Setup` object

The following instantiates the `phantomsetup.Setup` object.

```python
setup = phantomsetup.Setup()
```

## Set attributes and add particles

### Prefix

Set the prefix.

```python
setup.prefix = prefix
```

### Units

Set units.

```python
setup.set_units(
    length=length_unit, mass=mass_unit, gravitational_constant_is_unity=True
)
```

### Equation of state

Set the equation of state. We get `polyk` from the aspect ratio parametrization.

```python
polyk = polyk_for_locally_isothermal_disc(
    q_index, reference_radius, aspect_ratio, stellar_mass, gravitational_constant
)

setup.set_equation_of_state(ieos=ieos, polyk=polyk)
```

### Viscosity

Set the numerical viscosity to Phantom disc viscosity.

```python
setup.set_dissipation(disc_viscosity=True, alpha=alpha_artificial)
```

### Star

Add a star at the origin.

```python
setup.add_sink(
    mass=stellar_mass,
    accretion_radius=stellar_accretion_radius,
    position=stellar_position,
    velocity=stellar_velocity,
)
```

### Disc

Add the disc around the star.

```python
setup.add_disc(
    particle_type=particle_type,
    number_of_particles=number_of_particles,
    disc_mass=disc_mass,
    density_distribution=density_distribution,
    radius_range=(radius_min, radius_max),
    q_index=q_index,
    aspect_ratio=aspect_ratio,
    reference_radius=reference_radius,
    stellar_mass=stellar_mass,
    gravitational_constant=gravitational_constant,
    args=(radius_critical, gamma),
)
```

### Planet

Add a planet in orbit around the star.

```python
setup.add_sink(
    mass=planet_mass,
    accretion_radius=planet_accretion_radius,
    position=planet_position,
    velocity=planet_velocity,
)
```

## Write to file

Write the temporary dump file with the initial conditions and the Phantom in file.

```python
setup.write_dump_file()
setup.write_in_file()
```

## Plot

Now we plot some quantities to see what we have set up.

First is the particles in the xy-plane. The sink particles are marked in red.

```python
fig, ax = plt.subplots()
ax.plot(setup.x[::10], setup.y[::10], 'k.', ms=0.5)
for sink in setup.sinks:
    ax.plot(sink.position[0], sink.position[1], 'ro')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_aspect('equal')
```

Next we plot the particles in the rz-plane.

```python
fig, ax = plt.subplots()
ax.plot(setup.R[::10], setup.z[::10], 'k.', ms=0.5)
ax.set_xlabel('$R$')
ax.set_ylabel('$z$')
ax.set_aspect('equal')
ax.set_ylim(bottom=2 * setup.z.min(), top=2 * setup.z.max())
```

Finally, we plot $v_{\phi}$ as a function of radius.

```python
fig, ax = plt.subplots()
ax.plot(setup.R[::10], setup.vphi[::10], 'k.', ms=0.5)
ax.set_xlabel('$R$')
ax.set_ylabel('$v_{\phi}$')
```
