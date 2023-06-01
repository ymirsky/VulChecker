from pathlib import Path, PurePath
from typing import Iterable, Tuple


def _parts_with_dot(p: PurePath) -> Tuple[str, ...]:
    if p.is_absolute():
        return p.parts
    else:
        return (".",) + p.parts


def common_ancestor(paths: Iterable[PurePath]) -> PurePath:
    """Compute the common ancestor of some paths.

    This is a text-based manipulation.
    No filesystem access will occur.
    If you have mixed relative and absolute paths,
    or paths containing ``..`` components,
    use :meth:`pathlib.Path.resolve` on each path first.
    If you don't,
    you'll likely get a :exc:`ValueError`
    because there is no common ancestor.

    :raises IndexError: if the input iterable is empty.
    :raises ValueError: if there is no common ancestor.

    """
    paths = iter(paths)
    try:
        common = _parts_with_dot(next(paths))
    except StopIteration:
        raise IndexError("Can't compute common ancestor of empty iterable.")
    for path in paths:
        for i, (left, right) in enumerate(zip(common, _parts_with_dot(path))):
            if left != right:
                common = common[:i]
                break
        else:
            # Referring to i here is OK since _parts_with_dot never returns empty.
            common = common[: i + 1]
        if not common:
            raise ValueError("Paths have no common ancestor.")
    return PurePath(*common)


def relative_to_with_parents(dest: Path, source: Path) -> Path:
    """Relative path potentially including leading .. components.

    This is a text-based manipulation.
    No filesystem access will occur.
    You need to pass absolute paths,
    consider calling :meth:`pathlib.Path.resolve` first.
    The ``source`` should be a directory, not a file.

    """
    ancestor = common_ancestor([dest, source])
    up_components = len(source.parts) - len(ancestor.parts)
    return Path(*[".."] * up_components) / dest.relative_to(ancestor)
