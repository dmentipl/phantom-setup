import io
import pathlib
import re

from setuptools import setup

version = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open('phantomsetup/__init__.py', encoding='utf_8_sig').read(),
).group(1)

long_description = (pathlib.Path(__file__).parent / 'README.md').read_text()

setup(
    name='phantomsetup',
    version=version,
    author='Daniel Mentiplay',
    packages=['phantomsetup'],
    url='http://github.com/dmentipl/phantom-setup',
    license='MIT',
    description='Phantom setup with Python and HDF5',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['h5py', 'numpy', 'phantomconfig', 'scipy', 'tomlkit'],
)
