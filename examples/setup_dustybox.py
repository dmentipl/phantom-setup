"""
Setup the DUSTYBOX test problem.

The dust and gas are co-located in a box with uniform density. There is
an initial uniform differential velocity between the dust and gas.

Daniel Mentiplay, 2019.
"""

import phantomsetup.setups.dustybox as dustybox

# ------------------------------------------------------------------------------------ #
# Get the default parameters
parameters = dustybox.get_parameters()

# ------------------------------------------------------------------------------------ #
# See what parameters are available
print(parameters)

# ------------------------------------------------------------------------------------ #
# Change some values
parameters.prefix = 'my-dustybox'
parameters.drag_method = 'Epstein/Stokes'
parameters.dust_to_gas_ratio = (0.01, 0.01, 0.01, 0.01)
parameters.grain_size = (0.01, 0.1, 1.0, 10.0)
parameters.grain_density = 3.0

# ------------------------------------------------------------------------------------ #
# Generate the setup
setup = dustybox.setup(parameters)

# ------------------------------------------------------------------------------------ #
# Write dump and in file
setup.write_dump_file()
setup.write_in_file()
