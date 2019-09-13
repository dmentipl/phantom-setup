"""
Setup a protoplanetary disc with dust and planets.

Daniel Mentiplay, 2019.
"""

import phantomsetup.setups.disc as disc

# ------------------------------------------------------------------------------------ #
# Get the default parameters
parameters = disc.Parameters()

# ------------------------------------------------------------------------------------ #
# Modify parameters
parameters.use_dust = True
parameters.grain_size = (0.01, 0.1, 1.0)

# ------------------------------------------------------------------------------------ #
# Write parameter file
filename = 'disc.toml'
header = 'Disc setup'
parameters.write_to_file(filename=filename, overwrite=True, header=header)

# ------------------------------------------------------------------------------------ #
# Generate the setup
setup = disc.setup(parameters)

# ------------------------------------------------------------------------------------ #
# Write dump and in file
setup.write_dump_file()
setup.write_in_file()
