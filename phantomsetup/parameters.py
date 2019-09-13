from __future__ import annotations

import dataclasses
import pathlib
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import tomlkit


@dataclass
class ParametersBase:
    """Setup parameters base class."""

    def check_consistency(self) -> None:
        pass

    def write_to_file(
        parameters: dataclass,
        filename: Union[str, Path],
        *,
        header: str = None,
        overwrite: bool = False
    ) -> None:
        """
        Write the parameters to TOML file.

        Parameters
        ----------
        parameters : dataclass
            The parameters dataclass to write to file.
        filename : str or Path
            The name of the file to write. Should have extension
            '.toml'.
        header : str
            A header written as a TOML comment at the top of the file.
        overwrite : bool, default=False
            Whether to overwrite if the file exists.
        """

        if not overwrite:
            if pathlib.Path(filename).exists():
                raise ValueError('file already exists, add overwrite=True to overwrite')

        document = tomlkit.document()

        if header is not None:
            header = textwrap.wrap(header, 70)
            for header_ in header:
                document.add(tomlkit.comment(header_))

        for param in dataclasses.fields(parameters):
            name = param.name
            description = textwrap.wrap(param.metadata['description'], 70)
            value = getattr(parameters, param.name)
            if isinstance(value, tuple):
                value = list(value)
            document.add(tomlkit.nl())
            for desc in description:
                document.add(tomlkit.comment(desc))
            document.add(name, value)

        with open(filename, 'w') as fp:
            fp.write(tomlkit.dumps(document))

        return

    @classmethod
    def get_parameters(cls, filename: Union[str, Path] = None) -> ParametersBase:
        """Get parameters from file or from defaults.

        Parameters
        ----------
        filename : str or Path
            Read parameters from file, or get defaults if filename is None.

        Returns
        -------
        Parameters
            The parameters as a Parameters dataclass object.
        """
        if filename is not None:
            return cls(**read_parameter_file(filename))
        return cls()


def read_parameter_file(filename: Union[str, Path]) -> dict:
    """
    Read parameters from TOML file.

    Parameters
    ----------
    filename : str or Path
        The name of the file to read. Should have extension '.toml'.

    Returns
    -------
    dict
        A dictionary representation of the parameters file.
    """

    if not pathlib.Path(filename).exists():
        raise ValueError('parameter file does not exist')

    with open(filename, 'r') as fp:
        t = tomlkit.loads(fp.read())

    d = dict()
    for key, val in t.items():
        if isinstance(val, list):
            val = tuple(val)
        d[key] = val

    return d
