import gzip
from pathlib import Path

import click


def read_all(paths):
    for path in paths:
        with smart_open(path, "r") as f:
            yield from f


def smart_open(path, *args, **kwargs):
    if path == "-":
        return click.open_file(path, *args, **kwargs)

    path = Path(path)
    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    return opener(path, *args, **kwargs)
