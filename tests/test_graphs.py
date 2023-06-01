"""Graph fixtures for tests.

This file contains a bunch of factory functions to generate graphs used
during the tests. They are organized into classes to keep related things
together, but no instances of these classes are expected.

"""

import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as S
from hypothesis_networkx import graph_builder

from hector_ml.features import (
    EDGE_FEATURES,
    Feature,
    FeatureKind,
    feature_set,
    node_features,
)
from hector_ml.graphs import (
    GraphDataset,
    bfs_with_depth,
    mean_field_from_graph,
    sink_graph,
)


def _edge_dict(g):
    kwargs = {"data": True}
    if g.is_multigraph():
        kwargs["keys"] = True

    if g.is_directed():
        to_key = tuple
    else:
        if g.is_multigraph():

            def to_key(x):
                u, v, k = x
                if v < u:
                    return v, u, k
                else:
                    return u, v, k

        else:

            def to_key(x):
                return tuple(sorted(x))

    return dict((to_key(k), d) for *k, d in g.edges(**kwargs))


def assert_same_graph(g1, g2):
    """Assert that two graphs are the same.

    Note that this differs from :func:`networkx.is_isomorphic` because
    this function requires the node IDs to be the same, too.

    """
    assert (
        g1.is_directed() == g2.is_directed()
    ), "Graphs must be both directed or both undirected."
    assert (
        g1.is_multigraph() == g2.is_multigraph()
    ), "Graphs must be both allow or both disallow parallel edges."
    assert g1.graph == g2.graph
    assert dict(g1.nodes(data=True)) == dict(g2.nodes(data=True))
    assert _edge_dict(g1) == _edge_dict(g2)


@given(graph_builder(graph_type=nx.DiGraph, min_nodes=1))
def test_bfs_with_depth(graph):
    start = next(iter(graph))
    bfs_depth = list(bfs_with_depth(graph, start))
    bfs = [start] + [v for u, v in nx.bfs_edges(graph, start)]
    depths = [nx.shortest_path_length(graph, start, n) for n in bfs]
    assert bfs_depth == list(zip(bfs, depths))


dummy_graphs = graph_builder(
    graph_type=nx.Graph,
    edge_data=S.just({"type": 0, "dtype": 0}),
    min_nodes=1,
    max_nodes=5,
)


@given(dummy_graphs)
def test_mean_field_from_graph(graph):
    indexes = {"type": {"control_flow": 0, "def_use": 1}}
    node_features = feature_set([], indexes)
    edge_features = feature_set([Feature("type", FeatureKind.categorical)], indexes)
    graph.graph["label"] = ""
    graph_structure = mean_field_from_graph(graph, node_features, edge_features)
    adj, inc = graph_structure.structure
    assert adj.todense().tolist() == nx.adjacency_matrix(graph).todense().tolist()
    assert inc.todense().tolist() == nx.incidence_matrix(graph).todense().tolist()


def test_graph_dataset():
    nf = feature_set(node_features(SmallGraph.INDEXES), SmallGraph.INDEXES)
    ef = feature_set(EDGE_FEATURES, SmallGraph.INDEXES)
    dataset = list(GraphDataset([SmallGraph.graph_features()], nf, ef))
    assert len(dataset) == len(SmallGraph.MANIFESTATION_NODES)


class SmallGraph:
    """A small graph, but with enough content to be interesting.

    It consists of two connected components, each with a manifestation
    point. One has a root cause and the other does not.

    """

    POSITIVE_LABEL = "P"
    INDEXES = {
        "operation": {"call": 0, "mul": 1, "cmp": 2},
        "function": {"fscanf": 0, "malloc": 1},
        "dtype": {"int": 0, "unsigned long": 1, "void *": 2, "void": 3},
        "tag": {"root_cause": 0, "manifestation": 1, "guard": 2},
        "type": {"def_use": 0, "control_flow": 1},
    }
    MANIFESTATION_NODES = [2, 4]
    SOURCE_LINES = {
        25: "void foo() {",
        26: "int n; unsigned long m; char *data;",
        27: 'fscanf(stdin, "%d", &n);',
        28: "m = n * 2;",
        29: "data = malloc(m);",
        30: "}",
        33: "void bar() {",
        34: "int *data; unsigned long n;",
        35: "n = 4 * sizeof(int);",
        36: "data = malloc(n);",
        37: "if (!data);",
        38: "}",
    }

    @staticmethod
    def raw():
        """Raw input format."""
        graph = nx.MultiDiGraph()
        graph.add_node(
            0,
            operation="call",
            function="fscanf",
            dtype="int",
            condition=False,
            tag=["root_cause"],
            containing_function="foo",
            filename="foo.c",
            line_number=27,
        )
        graph.add_node(
            1,
            operation="mul",
            dtype="unsigned long",
            condition=False,
            tag=[],
            containing_function="foo",
            filename="foo.c",
            line_number=28,
        )
        graph.add_node(
            2,
            operation="call",
            function="malloc",
            dtype="void *",
            condition=False,
            tag=["manifestation"],
            label="P",
            containing_function="foo",
            filename="foo.c",
            line_number=29,
        )
        graph.add_node(
            3,
            operation="mul",
            dtype="unsigned long",
            condition=False,
            tag=[],
            containing_function="bar",
            filename="foo.c",
            line_number=35,
        )
        graph.add_node(
            4,
            operation="call",
            function="malloc",
            dtype="void *",
            condition=False,
            tag=["manifestation"],
            label="N",
            containing_function="bar",
            filename="foo.c",
            line_number=36,
        )
        graph.add_node(
            5,
            operation="cmp",
            dtype="void *",
            condition=True,
            tag=["guard"],
            containing_function="bar",
            filename="foo.c",
            line_number=37,
        )
        graph.add_edge(0, 1, type="def_use", dtype="int")
        graph.add_edge(0, 1, type="control_flow", dtype="void")
        graph.add_edge(1, 2, type="def_use", dtype="unsigned long")
        graph.add_edge(1, 2, type="control_flow", dtype="void")
        graph.add_edge(2, 3, type="control_flow", dtype="void")
        graph.add_edge(3, 4, type="def_use", dtype="unsigned long")
        graph.add_edge(3, 4, type="control_flow", dtype="void")
        graph.add_edge(4, 5, type="def_use", dtype="void *")
        graph.add_edge(4, 5, type="control_flow", dtype="void")
        return graph

    @staticmethod
    def merged_edges():
        """Raw input format with merged edges."""
        graph = nx.DiGraph()
        graph.add_node(
            0,
            operation="call",
            function="fscanf",
            dtype="int",
            condition=False,
            tag=["root_cause"],
            containing_function="foo",
            filename="foo.c",
            line_number=27,
        )
        graph.add_node(
            1,
            operation="mul",
            dtype="unsigned long",
            condition=False,
            tag=[],
            containing_function="foo",
            filename="foo.c",
            line_number=28,
        )
        graph.add_node(
            2,
            operation="call",
            function="malloc",
            dtype="void *",
            condition=False,
            tag=["manifestation"],
            label="P",
            containing_function="foo",
            filename="foo.c",
            line_number=29,
        )
        graph.add_node(
            3,
            operation="mul",
            dtype="unsigned long",
            condition=False,
            tag=[],
            containing_function="bar",
            filename="foo.c",
            line_number=35,
        )
        graph.add_node(
            4,
            operation="call",
            function="malloc",
            dtype="void *",
            condition=False,
            tag=["manifestation"],
            label="N",
            containing_function="bar",
            filename="foo.c",
            line_number=36,
        )
        graph.add_node(
            5,
            operation="cmp",
            dtype="void *",
            condition=True,
            tag=["guard"],
            containing_function="bar",
            filename="foo.c",
            line_number=37,
        )
        graph.add_edge(0, 1, type=["control_flow", "def_use"], dtype="int")
        graph.add_edge(1, 2, type=["control_flow", "def_use"], dtype="unsigned long")
        graph.add_edge(2, 3, type=["control_flow"], dtype="void")
        graph.add_edge(3, 4, type=["control_flow", "def_use"], dtype="unsigned long")
        graph.add_edge(4, 5, type=["control_flow", "def_use"], dtype="void *")
        return graph

    @staticmethod
    def categorical():
        """Categorical indexes format.

        The indexes used are SmallGraph.INDEXES.

        """
        graph = nx.DiGraph()
        graph.add_node(
            0,
            operation=0,
            function=0,
            dtype=0,
            condition=False,
            tag=[0],
            containing_function="foo",
            filename="foo.c",
            line_number=27,
        )
        graph.add_node(
            1,
            operation=1,
            dtype=1,
            condition=False,
            tag=[],
            containing_function="foo",
            filename="foo.c",
            line_number=28,
        )
        graph.add_node(
            2,
            operation=0,
            function=1,
            dtype=2,
            condition=False,
            tag=[1],
            label=1,
            containing_function="foo",
            filename="foo.c",
            line_number=29,
        )
        graph.add_node(
            3,
            operation=1,
            dtype=1,
            condition=False,
            tag=[],
            containing_function="bar",
            filename="foo.c",
            line_number=35,
        )
        graph.add_node(
            4,
            operation=0,
            function=1,
            dtype=2,
            condition=False,
            tag=[1],
            label=0,
            containing_function="bar",
            filename="foo.c",
            line_number=36,
        )
        graph.add_node(
            5,
            operation=2,
            dtype=2,
            condition=True,
            tag=[2],
            containing_function="bar",
            filename="foo.c",
            line_number=37,
        )
        graph.add_edge(0, 1, type=[1, 0], dtype=0)
        graph.add_edge(1, 2, type=[1, 0], dtype=1)
        graph.add_edge(2, 3, type=[1], dtype=3)
        graph.add_edge(3, 4, type=[1, 0], dtype=1)
        graph.add_edge(4, 5, type=[1, 0], dtype=2)
        return graph

    @staticmethod
    def graph_features():
        """With whole-graph features."""
        graph = nx.DiGraph(manifestation_nodes=SmallGraph.MANIFESTATION_NODES)
        graph.add_node(
            0,
            operation=0,
            function=0,
            dtype=0,
            condition=False,
            tag=[0],
            distance_manifestation=30,
            control_flow_out_degree=1,
            def_use_out_degree=1,
            betweenness=0.0,
            distance_root_cause=0,
            nearest_root_cause_op=0,
            containing_function="foo",
            filename="foo.c",
            line_number=27,
        )
        graph.add_node(
            1,
            operation=1,
            dtype=1,
            condition=False,
            tag=[],
            distance_manifestation=30,
            control_flow_out_degree=1,
            def_use_out_degree=1,
            betweenness=1 / 2,
            distance_root_cause=1,
            nearest_root_cause_op=0,
            containing_function="foo",
            filename="foo.c",
            line_number=28,
        )
        graph.add_node(
            2,
            operation=0,
            function=1,
            dtype=2,
            condition=False,
            tag=[1],
            distance_manifestation=30,
            control_flow_out_degree=1,
            def_use_out_degree=0,
            betweenness=0.0,
            distance_root_cause=2,
            nearest_root_cause_op=0,
            label=1,
            containing_function="foo",
            filename="foo.c",
            line_number=29,
        )
        graph.add_node(
            3,
            operation=1,
            dtype=1,
            condition=False,
            tag=[],
            distance_manifestation=30,
            control_flow_out_degree=1,
            def_use_out_degree=1,
            betweenness=0.0,
            distance_root_cause=30,
            nearest_root_cause_op=None,
            containing_function="bar",
            filename="foo.c",
            line_number=35,
        )
        graph.add_node(
            4,
            operation=0,
            function=1,
            dtype=2,
            condition=False,
            tag=[1],
            distance_manifestation=30,
            control_flow_out_degree=1,
            def_use_out_degree=1,
            betweenness=1 / 2,
            distance_root_cause=30,
            nearest_root_cause_op=None,
            label=0,
            containing_function="bar",
            filename="foo.c",
            line_number=36,
        )
        graph.add_node(
            5,
            operation=2,
            dtype=2,
            condition=True,
            tag=[2],
            distance_manifestation=30,
            control_flow_out_degree=0,
            def_use_out_degree=0,
            betweenness=0,
            distance_root_cause=30,
            nearest_root_cause_op=None,
            containing_function="bar",
            filename="foo.c",
            line_number=37,
        )
        graph.add_edge(0, 1, type=[1, 0], dtype=0)
        graph.add_edge(1, 2, type=[1, 0], dtype=1)
        graph.add_edge(2, 3, type=[1], dtype=3)
        graph.add_edge(3, 4, type=[1, 0], dtype=1)
        graph.add_edge(4, 5, type=[1, 0], dtype=2)
        return graph

    @staticmethod
    def _manifestation_2():
        graph = nx.Graph(
            filename="foo.c",
            containing_function="foo",
            line_number=29,
            label=1,
            manifestation_nodes=SmallGraph.MANIFESTATION_NODES,
        )
        graph.add_node(
            2,
            operation=0,
            function=None,
            dtype=2,
            condition=False,
            tag=[1],
            distance_manifestation=0,
            control_flow_out_degree=1,
            def_use_out_degree=0,
            betweenness=0.0,
            distance_root_cause=2,
            nearest_root_cause_op=0,
            containing_function="foo",
            filename="foo.c",
            line_number=29,
        )
        graph.add_node(
            0,
            operation=0,
            function=0,
            dtype=0,
            condition=False,
            tag=[0],
            distance_manifestation=2,
            control_flow_out_degree=1,
            def_use_out_degree=1,
            betweenness=0.0,
            distance_root_cause=0,
            nearest_root_cause_op=0,
            containing_function="foo",
            filename="foo.c",
            line_number=27,
        )
        graph.add_node(
            1,
            operation=1,
            dtype=1,
            condition=False,
            tag=[],
            distance_manifestation=1,
            control_flow_out_degree=1,
            def_use_out_degree=1,
            betweenness=1 / 2,
            distance_root_cause=1,
            nearest_root_cause_op=0,
            containing_function="foo",
            filename="foo.c",
            line_number=28,
        )
        graph.add_edge(0, 1, type=[1, 0], dtype=0)
        graph.add_edge(1, 2, type=[1, 0], dtype=1)
        return graph

    @staticmethod
    def _manifestation_4():
        graph = nx.Graph(
            filename="foo.c",
            containing_function="bar",
            line_number=36,
            label=0,
            manifestation_nodes=SmallGraph.MANIFESTATION_NODES,
        )
        graph.add_node(
            4,
            operation=0,
            function=None,
            dtype=2,
            condition=False,
            tag=[1],
            control_flow_out_degree=1,
            distance_manifestation=0,
            def_use_out_degree=1,
            betweenness=1 / 2,
            distance_root_cause=30,
            nearest_root_cause_op=None,
            containing_function="bar",
            filename="foo.c",
            line_number=36,
        )
        graph.add_node(
            0,
            operation=0,
            function=0,
            dtype=0,
            condition=False,
            tag=[0],
            distance_manifestation=4,
            control_flow_out_degree=1,
            def_use_out_degree=1,
            betweenness=0.0,
            distance_root_cause=0,
            nearest_root_cause_op=0,
            containing_function="foo",
            filename="foo.c",
            line_number=27,
        )
        graph.add_node(
            1,
            operation=1,
            dtype=1,
            condition=False,
            tag=[],
            distance_manifestation=3,
            control_flow_out_degree=1,
            def_use_out_degree=1,
            betweenness=1 / 2,
            distance_root_cause=1,
            nearest_root_cause_op=0,
            containing_function="foo",
            filename="foo.c",
            line_number=28,
        )
        graph.add_node(
            2,
            operation=0,
            function=1,
            dtype=2,
            condition=False,
            tag=[1],
            distance_manifestation=2,
            control_flow_out_degree=1,
            def_use_out_degree=0,
            betweenness=0.0,
            distance_root_cause=2,
            nearest_root_cause_op=0,
            containing_function="foo",
            filename="foo.c",
            line_number=29,
        )
        graph.add_node(
            3,
            operation=1,
            dtype=1,
            condition=False,
            tag=[],
            distance_manifestation=1,
            control_flow_out_degree=1,
            def_use_out_degree=1,
            betweenness=0.0,
            distance_root_cause=30,
            nearest_root_cause_op=None,
            containing_function="bar",
            filename="foo.c",
            line_number=35,
        )
        graph.add_edge(0, 1, type=[1, 0], dtype=0)
        graph.add_edge(1, 2, type=[1, 0], dtype=1)
        graph.add_edge(2, 3, type=[1], dtype=3)
        graph.add_edge(3, 4, type=[1, 0], dtype=1)
        return graph

    @classmethod
    def manifestation(cls, node):
        return getattr(cls, f"_manifestation_{node}")()


@pytest.mark.parametrize("node", SmallGraph.MANIFESTATION_NODES)
def test_small_graph_manifestation_first(node):
    graph = SmallGraph.manifestation(node)
    assert next(iter(graph)) == node


@pytest.mark.parametrize(
    "pivot,expected",
    [(n, SmallGraph.manifestation(n)) for n in SmallGraph.MANIFESTATION_NODES],
)
def test_sink_graph(pivot, expected):
    graph = SmallGraph.graph_features()
    output = sink_graph(graph, pivot, None)
    # Consumers should ignore the label, if it exists, but we don't
    # clean it up in the main implementation for performance.
    for _n, d in output.nodes(data=True):
        d.pop("label", None)
    assert_same_graph(output, expected)
    assert_same_graph(graph, SmallGraph.graph_features())

    # The manifestation point must be the first node in iteration order
    first_node = next(iter(output))
    assert output.nodes[first_node]["distance_manifestation"] == 0
