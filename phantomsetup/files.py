import dataclasses
import pathlib
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import tomlkit


def write_parameter_file(
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
        document.add(tomlkit.comment(header))

    for param in dataclasses.fields(parameters):
        name = param.name
        comment = textwrap.wrap(param.metadata['description'], 70)
        value = getattr(parameters, param.name)
        if isinstance(value, tuple):
            value = list(value)
        document.add(tomlkit.nl())
        for comment_ in comment:
            document.add(tomlkit.comment(comment_))
        document.add(name, value)

    with open(filename, 'w') as fp:
        fp.write(tomlkit.dumps(document))

    return
