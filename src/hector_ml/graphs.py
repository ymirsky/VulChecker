import json

import networkx as nx
from torch.utils.data import IterableDataset

from hector_ml._graphs import mean_field_from_node_link_data


def bfs_with_depth(graph, start, reverse=False, depth_limit=None):
    """Get nodes in breadth-first order, with depths."""
    depths = {start: 0}
    yield start, 0
    for u, v in nx.bfs_edges(graph, start, reverse, depth_limit):
        depth = depths[v] = depths[u] + 1
        yield v, depth


def mean_field_from_graph(graph, node_features, edge_features):
    return mean_field_from_node_link_data(
        nx.node_link_data(graph), node_features, edge_features
    )


class JSONGraphs:
    def __init__(self, graphs_file):
        self.graphs_file = graphs_file

    def __iter__(self):
        self.graphs_file.seek(0)
        for line in self.graphs_file:
            yield nx.node_link_graph(json.loads(line))


NODE_METADATA = ["filename", "line_number", "containing_function", "label"]


def sink_node_link_data(graph, pivot, depth_limit):
    """Build a graph observation for a particular manifestation point.

    For machine learning purposes, we create multiple copies of the PDG
    with features based on different manifestation points. For each of
    these graphs, the graph label is taken from the label of the chosen
    manifestation point.

    """
    graph_info = graph.graph.copy()
    for k in NODE_METADATA:
        graph_info[k] = graph.nodes[pivot].get(k)
    nodes = []
    node_set = set()
    for node, depth in bfs_with_depth(
        graph, pivot, reverse=True, depth_limit=depth_limit
    ):
        node_set.add(node)
        new_node_data = dict(graph.nodes[node], distance_manifestation=depth, id=node)
        if node == pivot:
            new_node_data["function"] = None
        nodes.append(new_node_data)
    edges = []
    for node in node_set:
        neighbors = graph[node]
        for neighbor, data in neighbors.items():
            if neighbor in node_set:
                edges.append(dict(data, source=node, target=neighbor))

    return {
        "directed": False,
        "multigraph": False,
        "graph": graph_info,
        "nodes": nodes,
        "links": edges,
    }


def sink_graph(graph, pivot, depth_limit):
    return nx.node_link_graph(sink_node_link_data(graph, pivot, depth_limit))


class GraphDataset(IterableDataset):
    def __init__(self, graphs, node_feature_set, edge_feature_set, depth_limit=None):
        self.graphs = graphs
        self.node_features = node_feature_set
        self.edge_features = edge_feature_set
        self.depth_limit = depth_limit

    def __iter__(self):
        for graph in self.graphs:
            for manifestation_node in graph.graph["manifestation_nodes"]:
                label = graph.nodes[manifestation_node]["label"]
                manif_nldata = sink_node_link_data(
                    graph, manifestation_node, self.depth_limit
                )
                yield (
                    mean_field_from_node_link_data(
                        manif_nldata, self.node_features, self.edge_features
                    ),
                    int(label),
                )


def sink_graph_infos(graphs):
    for graph in graphs:
        for manifestation_node in graph.graph["manifestation_nodes"]:
            node_info = graph.nodes[manifestation_node]
            node_metadata = {k: node_info.get(k) for k in NODE_METADATA}
            yield dict(graph.graph, **node_metadata)
