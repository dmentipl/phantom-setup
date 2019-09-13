"""
Setup a protoplanetary disc with dust and planets.

Daniel Mentiplay, 2019.
"""

import phantomsetup.setups.disc as disc

# ------------------------------------------------------------------------------------ #
# Get the default parameters
parameters = disc.Parameters()

# ------------------------------------------------------------------------------------ #
# Generate the setup
setup = disc.setup(parameters)
