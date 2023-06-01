from inspect import isclass

import numpy as np
import pytest

from hector_ml.features import Feature, FeatureKind, feature_set


def is_exception(thing):
    return isinstance(thing, BaseException) or (
        isclass(thing) and issubclass(thing, BaseException)
    )


@pytest.mark.parametrize(
    "data,expected",
    [
        ({"foo": 1, "bar": 3, "baz": [0, 1], "qux": "24"}, [0, 1, 3, 1, 1, 24, 0]),
        ({"foo": None, "bar": 1, "baz": [], "qux": "0"}, [0, 0, 1, 0, 0, 0, 0]),
        ({"foo": None, "bar": 1, "baz": None, "qux": "none"}, [0, 0, 1, 0, 0, 0, 1]),
        ({"foo": None, "bar": 1, "qux": "none"}, [0, 0, 1, 0, 0, 0, 1]),
        ({"bar": 1, "baz": [], "qux": "0"}, [0, 0, 1, 0, 0, 0, 0]),
        ({"foo": None, "bar": 1, "baz": None}, [0, 0, 1, 0, 0, 0, 1]),
        ({"foo": 0, "bar": None, "baz": [], "qux": None}, TypeError),
        ({}, KeyError),
    ],
)
def test_feature_row(data, expected):
    features = [
        Feature("foo", FeatureKind.categorical),
        Feature("bar"),
        Feature("baz", FeatureKind.categorical_set),
        Feature("qux", FeatureKind.optional_numeric),
    ]
    categorical_counts = {"foo": 2, "baz": 2}
    indexes = {k: range(v) for k, v in categorical_counts.items()}
    features = feature_set(features, indexes)
    feature_matrix = np.zeros((1, features.total_width), dtype=np.float32)
    if is_exception(expected):
        with pytest.raises(expected):
            features.feature_row(feature_matrix, data, 0)
    else:
        features.feature_row(feature_matrix, data, 0)
        assert feature_matrix.tolist() == [expected]
