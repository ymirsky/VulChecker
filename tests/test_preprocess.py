import json
import random
from collections import Counter
from pathlib import PurePath

import networkx as nx
import pytest

from hector_ml.features import Feature, FeatureKind
from hector_ml.preprocess import (
    add_invariant_graph_features,
    component_information,
    merge_edges,
    parallel_root_cause_distances,
    prepare_indexes_for_training,
    read_graph_file,
    relativize_file_names,
    remove_llvm_internal_functions,
    resolve_root_cause_op,
    translate_categorical_features,
    translate_item_categorical_features,
    translate_manifestation_point_labels,
)

from .test_graphs import SmallGraph, assert_same_graph


@pytest.fixture()
def fixed_randomness():
    state = random.getstate()
    random.seed(0)
    yield
    random.setstate(state)


@pytest.mark.parametrize(
    "ops,final",
    [
        ([], None),
        ([1], 1),
        ([1, 1, 1], 1),
        ([1, 1, 2], 1),
        ([0, 1, 1], 0),
        ([1, 2, 3], 3),
    ],
)
def test_resolve_root_cause_op(fixed_randomness, ops, final):
    data = {"root_cause_ops": Counter(ops)}
    resolve_root_cause_op(data, 0)
    assert data["nearest_root_cause_op"] == final


def test_prepare_indexes_empty():
    indexes = prepare_indexes_for_training({})
    assert indexes["foo"]["a"] == 0
    assert indexes["foo"]["b"] == 1


def test_prepare_indexes_partial():
    indexes = prepare_indexes_for_training({"foo": {"a": 0, "b": 1}})
    assert indexes["foo"]["c"] == 2
    assert indexes["foo"]["a"] == 0
    assert indexes["foo"]["b"] == 1


def test_read_graph_file():
    example_graph_json = """
    {
        "graph": {},
        "nodes": [
            {
                "id": 0,

                "static_value": null,
                "operation": "add",
                "function": null,
                "dtype": "int64",
                "condition": false,
                "tag": "other",
                "file": "foo.c",
                "line_number": 27,
                "containing_function": "foo",
                "label": "negative"
            }
        ],
        "links": [
            {
                "source": 0,
                "target": 0,

                "type": "def_use",
                "dtype": "int64"
            }
        ]
    }
    """

    example_graph = nx.MultiDiGraph()
    example_graph.add_node(
        0,
        static_value=None,
        operation="add",
        function=None,
        dtype="int64",
        condition=False,
        tag="other",
        file="foo.c",
        line_number=27,
        containing_function="foo",
        label="negative",
    )
    example_graph.add_edge(0, 0, type="def_use", dtype="int64")

    graphs = list(read_graph_file([example_graph_json]))
    assert_same_graph(graphs[0], example_graph)


def test_merge_edges():
    graph = SmallGraph.raw()
    new_graph = merge_edges(graph)
    assert_same_graph(new_graph, SmallGraph.merged_edges())


def test_translate_item_categorical_features():
    features = [
        Feature("foo", FeatureKind.categorical),
        Feature("other_foo", FeatureKind.categorical, "foo"),
        Feature("bar", FeatureKind.categorical),
        Feature("baz", FeatureKind.categorical_set),
    ]
    data = {"foo": "a", "other_foo": "b", "baz": ["c", "d"], "qux": "e"}
    expected_data = {"foo": 0, "other_foo": 1, "baz": [0, 1], "qux": "e"}

    indexes = prepare_indexes_for_training({})
    translate_item_categorical_features(data, features, indexes)
    assert data == expected_data


def test_translate_item_missing_categorical_features():
    features = [
        Feature("foo", FeatureKind.categorical),
        Feature("other_foo", FeatureKind.categorical, "foo"),
        Feature("some_foos", FeatureKind.categorical_set, "foo"),
    ]
    data = {"foo": "a", "other_foo": "b", "some_foos": ["c", "d"]}
    expected_data = {"foo": None, "other_foo": 0, "some_foos": [1]}
    indexes = {"foo": {"b": 0, "c": 1}}

    translate_item_categorical_features(data, features, indexes)
    assert data == expected_data


def test_translate_categorical_features():
    graph = SmallGraph.merged_edges()
    translate_manifestation_point_labels(graph, SmallGraph.POSITIVE_LABEL)
    translate_categorical_features(graph, SmallGraph.INDEXES)
    assert_same_graph(graph, SmallGraph.categorical())


def test_save_load_indexes_non_string():
    features = [Feature("foo", FeatureKind.categorical)]
    indexes = prepare_indexes_for_training({})

    data = {"foo": True}
    translate_item_categorical_features(data, features, indexes)
    assert data["foo"] == 0

    indexes = prepare_indexes_for_training(json.loads(json.dumps(indexes)))

    data = {"foo": True}
    translate_item_categorical_features(data, features, indexes)
    assert data["foo"] == 0


def test_add_invariant_graph_features():
    graph = SmallGraph.categorical()
    add_invariant_graph_features(
        graph,
        call_index=SmallGraph.INDEXES["operation"]["call"],
        root_cause_index=SmallGraph.INDEXES["tag"]["root_cause"],
        manifestation_index=SmallGraph.INDEXES["tag"]["manifestation"],
        edge_type_indexes=SmallGraph.INDEXES["type"],
    )
    assert graph.graph["manifestation_nodes"] == SmallGraph.MANIFESTATION_NODES


def test_tiny_betweenness():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    betweenness, diameter = component_information(graph)
    assert betweenness == {0: 0.0, 1: 0.0}
    assert diameter == {0: 2, 1: 2}


def test_parallel_root_cause_distances():
    graph = nx.DiGraph()
    graph.add_node(0, operation="foo", root_cause_ops=Counter())
    graph.add_node(1, root_cause_ops=Counter())
    graph.add_node(2, root_cause_ops=Counter())
    graph.add_node(3, operation="bar", root_cause_ops=Counter())
    graph.add_node(4, root_cause_ops=Counter())
    graph.add_node(5, operation="baz", root_cause_ops=Counter())
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(3, 4)
    graph.add_edge(4, 2)
    graph.add_edge(5, 3)
    parallel_root_cause_distances(graph, [0, 3, 5])

    expected = nx.DiGraph()
    expected.add_node(
        0, operation="foo", distance_root_cause=0, root_cause_ops=Counter(["foo"])
    )
    expected.add_node(1, distance_root_cause=1, root_cause_ops=Counter(["foo"]))
    expected.add_node(2, distance_root_cause=2, root_cause_ops=Counter(["foo", "bar"]))
    expected.add_node(
        3, operation="bar", distance_root_cause=0, root_cause_ops=Counter(["bar"])
    )
    expected.add_node(4, distance_root_cause=1, root_cause_ops=Counter(["bar"]))
    expected.add_node(
        5, operation="baz", distance_root_cause=0, root_cause_ops=Counter(["baz"])
    )
    expected.add_edge(0, 1)
    expected.add_edge(1, 2)
    expected.add_edge(3, 4)
    expected.add_edge(4, 2)
    expected.add_edge(5, 3)

    assert_same_graph(graph, expected)


def test_remove_llvm_internal_functions():
    graph = nx.DiGraph()
    graph.add_node(0, function=None)
    graph.add_node(1, function=None)
    graph.add_node(2, function="llvm.dbg.define")
    graph.add_node(3, function="llvm.load.relative")
    graph.add_node(4, function="printf")
    graph.add_edge(0, 2, type="def_use")
    graph.add_edge(1, 2, type="control_flow")
    graph.add_edge(2, 3, type="control_flow")
    graph.add_edge(3, 4, type="control_flow")
    remove_llvm_internal_functions(graph)

    expected = nx.DiGraph()
    expected.add_node(0, function=None)
    expected.add_node(1, function=None)
    expected.add_node(3, function="llvm.load.relative")
    expected.add_node(4, function="printf")
    expected.add_edge(1, 3, type="control_flow")
    expected.add_edge(3, 4, type="control_flow")

    assert_same_graph(graph, expected)


@pytest.mark.parametrize("stem", [PurePath(), PurePath("foo")])
def test_relativize_filenames(stem):
    g = nx.DiGraph()
    g.add_node(0, filename="")
    g.add_node(1, filename=str(stem / "libfoo/foo.c"))
    g.add_node(2, filename=str(stem / "util.c"))
    g.add_node(3)
    relativize_file_names(g)
    assert dict(g.nodes(data="filename")) == {
        0: "",
        1: "libfoo/foo.c",
        2: "util.c",
        3: None,
    }


@pytest.mark.parametrize("stem", [PurePath(), PurePath("foo")])
def test_relativize_filenames_all_alike(stem):
    g = nx.DiGraph()
    g.add_node(0, filename="")
    g.add_node(1, filename=str(stem / "foo.c"))
    g.add_node(2, filename=str(stem / "foo.c"))
    g.add_node(3)
    relativize_file_names(g)
    assert dict(g.nodes(data="filename")) == {
        0: "",
        1: "foo.c",
        2: "foo.c",
        3: None,
    }


@pytest.mark.parametrize("stem", [PurePath(), PurePath("foo")])
def test_relativize_filenames_explicit_source(stem):
    g = nx.DiGraph()
    g.add_node(0, filename="")
    g.add_node(1, filename=str(stem / "foo.c"))
    g.add_node(2, filename=str(stem / "foo.c"))
    g.add_node(3)
    g.add_node(4, filename="/usr/include/c++/9/vector")
    relativize_file_names(g, stem)
    assert dict(g.nodes(data="filename")) == {
        0: "",
        1: "foo.c",
        2: "foo.c",
        3: None,
        4: "/usr/include/c++/9/vector",
    }
