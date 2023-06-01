import pathlib
from typing import Sequence

import attr


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SourceFile:
    path: pathlib.Path
    extra_flags: Sequence[str] = attr.Factory(tuple)
