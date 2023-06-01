from pathlib import PurePath

import pytest
from hypothesis import given, note
from hypothesis import strategies as S

from hector_ml.paths import common_ancestor


def paths(
    allow_absolute=True,
    allow_relative=True,
    allow_parent=True,
    label_characters=S.characters(
        blacklist_characters=["/", "\0"], blacklist_categories=["Cs"]
    ),
    max_label_length=16,
    max_path_length=8,
):
    roots = []
    if allow_absolute:
        roots.append("/")
    if allow_relative:
        roots.append(".")
    if not roots:
        raise ValueError(
            "At least one of allow_absolute and allow_relative must be True."
        )

    heads = S.sampled_from(roots)
    labels = S.text(label_characters, min_size=1, max_size=max_label_length).filter(
        lambda x: x not in (".", "..")
    )
    if allow_parent:
        labels = S.one_of(labels, S.just(".."))
    tails = S.lists(labels, max_size=max_path_length)

    return S.builds(lambda h, t: PurePath(h, *t), heads, tails)


def is_relative_to(path, *other):
    try:
        path.relative_to(*other)
    except ValueError:
        return False
    else:
        return True


@given(S.lists(paths(allow_relative=False, allow_parent=False), min_size=1, max_size=8))
def test_common_ancestor_absolute(paths):
    ancestor = common_ancestor(paths)
    note(ancestor)
    assert all([is_relative_to(p, ancestor) for p in paths])


@given(S.lists(paths(allow_absolute=False, allow_parent=False), min_size=1, max_size=8))
def test_common_ancestor_relative(paths):
    ancestor = common_ancestor(paths)
    note(ancestor)
    assert all([is_relative_to(p, ancestor) for p in paths])


def test_common_ancestor_empty():
    with pytest.raises(IndexError):
        common_ancestor([])


def test_common_ancestor_no_common_ancestor():
    with pytest.raises(ValueError):
        common_ancestor([PurePath("/"), PurePath(".")])
