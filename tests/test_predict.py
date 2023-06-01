from collections import defaultdict

import networkx as nx
import numpy as np
import pytest
from more_itertools import one

from hector_ml import predict
from hector_ml.model import Predictor
from tests.test_graphs import SmallGraph, assert_same_graph


def test_split_graph():
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(2, 4)
    g.add_edge(2, 5)
    g.add_edge(3, 6)

    expected = nx.Graph()
    expected.add_edge(0, 1)
    expected.add_edge(0, 3)
    expected.add_edge(3, 6)

    actual = predict.split_graph(g, view_node=0, remove_node=2)
    assert_same_graph(expected, actual)


@pytest.mark.parametrize("manifestation_node", SmallGraph.MANIFESTATION_NODES)
def test_predict_root_cause(manifestation_node):
    def fake_model(dataset):
        return np.random.normal(size=(dataset.structure[0].shape[0], 2))

    mock_predictor = Predictor(
        model=fake_model,
        model_params={"n_classes": 2},
        indexes=SmallGraph.INDEXES,
        reduction_mode="sum",
        cwe=0,
    )
    raw_graph = SmallGraph.raw()
    nodes_by_tag = defaultdict(set)
    for n, tag in raw_graph.nodes(data="tag"):
        for t in tag:
            nodes_by_tag[t].add(n)

    graph = SmallGraph.manifestation(manifestation_node)
    result = predict.predict_root_cause(graph, mock_predictor)

    available_root_causes = set(graph) & nodes_by_tag["root_cause"]
    if available_root_causes:
        found_root_cause = one(
            n
            for n, d in SmallGraph.raw().nodes(data=True)
            if predict.Location.from_node_data(d) == result
        )
        assert found_root_cause in available_root_causes
    else:
        assert result is None


def test_find_vulns():
    def fake_model(dataset):
        return np.matlib.repmat(
            np.array([-1, 1], dtype=np.float32), dataset.structure[0].shape[0], 1
        )

    mock_predictor = Predictor(
        model=fake_model,
        model_params={"n_classes": 2},
        indexes=SmallGraph.INDEXES,
        reduction_mode="sum",
        cwe=0,
    )

    graphs = [SmallGraph.manifestation(n) for n in SmallGraph.MANIFESTATION_NODES]
    # assert no exceptions
    predict.find_vulns(graphs, mock_predictor)
