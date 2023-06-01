import operator
from typing import Iterable, Optional

import attr
import click
import networkx as nx
import numpy as np
import torch

from hector_ml.click_helpers import smart_open
from hector_ml.graphs import JSONGraphs, sink_graph
from hector_ml.model import PredictionStyle, Predictor


@attr.s(auto_attribs=True, slots=True, frozen=True)
class Location:
    file_name: str
    line_number: int

    @classmethod
    def from_node_data(cls, data):
        return cls(file_name=data["filename"], line_number=data["line_number"])


@attr.s(auto_attribs=True, slots=True, frozen=True)
class Prediction:
    manifestation_point: Location
    root_cause: Optional[Location]
    probability: float


def split_graph(graph: nx.Graph, view_node, remove_node) -> nx.Graph:
    smaller = nx.subgraph_view(graph, filter_node=nx.filters.hide_nodes([remove_node]))
    return graph.subgraph(nx.node_connected_component(smaller, view_node))


def predict_root_cause(graph: nx.Graph, predictor: Predictor) -> Location:
    root_cause_index = predictor.indexes["tag"]["root_cause"]
    manifestation_node = next(iter(graph))

    stripped_graphs = []
    root_cause_locations = []
    for n, d in graph.nodes(data=True):
        if root_cause_index in d["tag"] and n != manifestation_node:
            root_cause_locations.append(Location.from_node_data(d))
            stripped_graphs.append(split_graph(graph, manifestation_node, n))

    if not stripped_graphs:
        return None

    stripped_probs = predictor.predict_graphs(stripped_graphs)[:, 1]
    return root_cause_locations[np.argmin(stripped_probs)]


def find_vulns(
    graph: nx.Graph, predictor: Predictor, threshold: float = 0.5, predicate=None
) -> Iterable[Prediction]:
    for pivot in graph.graph["manifestation_nodes"]:
        if predicate is None or predicate(graph.nodes[pivot]):
            sg = sink_graph(graph, pivot, predictor.depth_limit)
            prob = predictor.predict_graphs([sg], style=PredictionStyle.probabilities)[
                0, 1
            ]
            if prob >= threshold:
                yield Prediction(
                    manifestation_point=Location.from_node_data(sg.graph),
                    root_cause=predict_root_cause(sg, predictor),
                    probability=prob,
                )


def file_line_predicate(filename, line_number):
    ref = {}
    if filename is not None:
        ref["filename"] = filename
    if line_number is not None:
        ref["line_number"] = line_number
    if not ref:
        return None
    key = operator.itemgetter(*ref)
    val = key(ref)

    def predicate(node_data, key=key, val=val):
        return key(node_data) == val

    return predicate


@click.command()
@click.option(
    "--device", type=torch.device, default="cuda", help="Device on which to run."
)
@click.option(
    "--batch-size",
    type=click.IntRange(1, None),
    default=64,
    show_default=True,
    help="Maximum number of graphs to process at once.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0, 1),
    default=0.5,
    show_default=True,
    help="Score threshold for detection.",
)
@click.option("--filename", help="Only predict nodes in this file.")
@click.option("--line-number", type=int, help="Only predict nodes on this line.")
@click.argument("model_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("graphs", type=click.Path(exists=True, dir_okay=False))
def main(device, batch_size, threshold, filename, line_number, model_dir, graphs):
    predictor = Predictor.from_checkpoint_dir(
        model_dir, map_location=device, batch_size=batch_size
    )
    mp_format = (
        f"{{0.file_name}}:{{0.line_number}}:W:CWE-{predictor.cwe} manifests here "
        f"with probability {{1:.3}}"
    )
    rc_format = (
        f"{{0.file_name}}:{{0.line_number}}:W:CWE-{predictor.cwe} root cause here"
    )
    predicate = file_line_predicate(filename, line_number)
    with smart_open(graphs) as f:
        graphs = JSONGraphs(f)
        for graph in graphs:
            for pred in find_vulns(graph, predictor, threshold, predicate):
                print(mp_format.format(pred.manifestation_point, pred.probability))
                if pred.root_cause:
                    print(rc_format.format(pred.root_cause))
