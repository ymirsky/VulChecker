from typing import Iterable

import attr

from hector_ml._features import FeatureKind, FeatureSet
from hector_ml.types import Indexes


@attr.s(frozen=True, slots=True, auto_attribs=True)
class Feature:
    """Represent a feature.

    :param name: The name of this feature.
    :param kind: The kind of this feature.
    :param dictionary:
       The name of the feature dictionary to use.
       Defaults to the same as ``name``.
       This field is only used if ``kind`` is not ``FeatureKind.numeric``.

    """

    name: str
    kind: FeatureKind = FeatureKind.numeric
    dictionary: str = attr.Factory(lambda self: self.name, takes_self=True)


def feature_set(features: Iterable[Feature], indexes: Indexes) -> FeatureSet:
    return FeatureSet.from_features_and_indexes(features, indexes)


# Features for HECTOR ##################################################

BASE_NODE_FEATURES = [
    Feature("static_value", FeatureKind.optional_numeric),
    Feature("operation", FeatureKind.categorical),
    Feature("function", FeatureKind.categorical),
    Feature("dtype", FeatureKind.categorical),
    Feature("condition"),
    Feature("betweenness"),
    Feature("distance_manifestation"),
    Feature("distance_root_cause"),
    Feature("nearest_root_cause_op", FeatureKind.categorical, "operation"),
    Feature("tag", FeatureKind.categorical_set),
]
EDGE_FEATURES = [
    Feature("type", FeatureKind.categorical_set),
    Feature("dtype", FeatureKind.categorical),
]


def node_features(indexes):
    yield from BASE_NODE_FEATURES
    for edge_type in indexes["type"]:
        yield Feature(f"{edge_type}_out_degree")
