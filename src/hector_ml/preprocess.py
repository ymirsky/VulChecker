import json
import random
from collections import Counter, defaultdict, deque
from functools import lru_cache
from itertools import count
from pathlib import PurePath
from typing import Optional

import click
import networkit as nk
import networkx as nx
from more_itertools import one

from hector_ml.click_helpers import read_all, smart_open
from hector_ml.features import BASE_NODE_FEATURES, EDGE_FEATURES, FeatureKind
from hector_ml.paths import common_ancestor
from hector_ml.types import Indexes

NODE_CATEGORICAL = [
    f
    for f in BASE_NODE_FEATURES
    if f.kind in (FeatureKind.categorical, FeatureKind.categorical_set)
]
EDGE_CATEGORICAL = [
    f
    for f in EDGE_FEATURES
    if f.kind in (FeatureKind.categorical, FeatureKind.categorical_set)
]

CALL_OPERATION = "call"
ROOT_CAUSE_TAG = "root_cause"
MANIFESTATION_POINT_TAG = "manifestation"

LARGE_GRAPH_SIZE = 100
INFINITY_MULTIPLIER = 10

POSITIVE_LABELS = {
    121: "stack_overflow",
    122: "heap_overflow",
    190: "overflowed_call",
    191: "underflowed_call",
    415: "second_free",
    416: "use_after_free",
}


def read_graph_file(fp):
    for line in fp:
        yield nx.node_link_graph(json.loads(line), directed=True)


def manifestation_point_label(data, positive_label):
    if MANIFESTATION_POINT_TAG not in data["tag"]:
        data.pop("label", None)
    else:
        data["label"] = any(lab == positive_label for lab in data["label"])


def translate_manifestation_point_labels(graph, positive_label):
    for _n, d in graph.nodes(data=True):
        manifestation_point_label(d, positive_label)


def _guess_source_dir(graph: nx.DiGraph) -> PurePath:
    nodes_by_filename = defaultdict(list)
    for n, d in graph.nodes(data="filename"):
        if d:
            nodes_by_filename[PurePath(d)].append(n)
    if len(nodes_by_filename) == 1:
        # Only one filename, so use the parent as the source_dir
        source_dir = next(iter(nodes_by_filename)).parent
    else:
        source_dir = common_ancestor(nodes_by_filename)
    return source_dir


def relativize_file_names(graph: nx.DiGraph, source_dir: PurePath = None):
    if source_dir is None:
        source_dir = _guess_source_dir(graph)
    if source_dir == PurePath():
        # source dir is current working directory, no modifications needed
        return

    @lru_cache(maxsize=None)
    def _relativize_file(fn: str) -> str:
        try:
            return str(PurePath(fn).relative_to(source_dir))
        except ValueError:
            return fn

    for n, d in graph.nodes(data=True):
        fn = d.get("filename")
        if fn:
            d["filename"] = _relativize_file(fn)


def merge_edges(graph: nx.MultiDiGraph) -> nx.DiGraph:
    etype = defaultdict(set)
    edtype = {}
    for u, v, d in graph.edges(data=True):
        etype[(u, v)].add(d["type"])
        if d["type"] == "def_use":
            edtype[(u, v)] = d["dtype"]

    merged = nx.DiGraph()
    merged.graph.update(graph.graph)
    merged.add_nodes_from(graph.nodes(data=True))
    merged.add_edges_from(
        (u, v, {"type": sorted(etype[(u, v)]), "dtype": edtype.get((u, v), "void")})
        for u, v in etype
    )
    return merged


def _get_index(index_lookup, value):
    """Get a value from the dictionary or None.

    This looks like it should be the same as

    ::

        index_lookup.get(value)

    but it differs if index_lookup is a defaultdict. This function will
    use the default factory for a defaultdict and fall back to None only
    for a regular dictionary.

    """
    try:
        return index_lookup[value]
    except KeyError:
        return None


def translate_item_categorical_features(data, features, indexes):
    for f in features:
        index_lookup = indexes[f.dictionary]
        if data.get(f.name) is not None:
            if f.kind == FeatureKind.categorical:
                data[f.name] = _get_index(index_lookup, str(data[f.name]))
            elif f.kind == FeatureKind.categorical_set:
                translated = []
                for v in data[f.name]:
                    index = _get_index(index_lookup, v)
                    if index is not None:
                        translated.append(index)
                data[f.name] = translated


def translate_categorical_features(graph, indexes):
    """Convert categorical features to category indexes.

    For a graphs, convert the node and edge data with names
    matching categorical features to contain category indexes instead of
    real values.

    :param graph: Graph to convert.
    :param indexes: Dictionary of feature dictionaries to use.

    """
    for _n, d in graph.nodes(data=True):
        translate_item_categorical_features(d, NODE_CATEGORICAL, indexes)
    for _u, _v, d in graph.edges(data=True):
        translate_item_categorical_features(d, EDGE_CATEGORICAL, indexes)


def resolve_root_cause_op(data, call_index):
    """Figure out the nearest root cause operation feature.

    There may be multiple "nearest" root cause nodes, so some logic is
    required to figure out what the final value should be. If any of the
    root cause nodes is a function call, then the nearest operation is
    declared to be "function call". Otherwise, the plurality operation
    is chosen. Otherwise, a random selection from among all the nodes is
    made.

    """
    popular_root_causes = data.pop("root_cause_ops")
    if call_index in popular_root_causes:
        # Function calls take priority
        data["nearest_root_cause_op"] = call_index
    else:
        popular = popular_root_causes.most_common(2)
        if len(popular) == 2 and popular[0][1] == popular[1][1]:
            # A tie. Select at random.
            data["nearest_root_cause_op"] = random.choices(
                list(popular_root_causes.keys()), popular_root_causes.values()
            )[0]
        elif popular:
            # A winner. Use it.
            data["nearest_root_cause_op"] = popular[0][0]
        else:
            # No root cause found.
            data["nearest_root_cause_op"] = None


def component_information(graph):
    betweenness = {}
    diameter = {}
    for component_nodes in nx.connected_components(graph.to_undirected(as_view=True)):
        component_nodes = list(component_nodes)
        component_size = len(component_nodes)
        diameter.update(dict.fromkeys(component_nodes, component_size))

        component = graph.subgraph(component_nodes)
        # If the component has two or fewer nodes,
        # there's a division by zero inside EstimateBetweenness.
        # To avoid that, just say all nodes have 0 centrality.
        if component_size > 2:
            component = nk.nxadapter.nx2nk(component)
            # NetworKit docs don't give any guidance on selecting nSamples,
            # so I just copied the value used in the example.
            # positional-only arguments: G, nSamples, normalized, parallel_flag
            centrality = nk.centrality.EstimateBetweenness(component, 50, True, True)
            centrality.run()
            betweenness.update(zip(component_nodes, centrality.scores()))
        else:
            betweenness.update(dict.fromkeys(component_nodes, 0.0))
    return betweenness, diameter


def parallel_root_cause_distances(graph, root_cause_nodes):
    depth = dict.fromkeys(root_cause_nodes, 0)
    to_process = deque()
    operations = graph.nodes("operation")
    for node in root_cause_nodes:
        to_process.append((node, operations[node]))
    while to_process:
        node, operation = to_process.popleft()
        this_depth = depth[node]
        data = graph.nodes[node]
        data["distance_root_cause"] = this_depth
        data["root_cause_ops"][operation] += 1
        next_depth = this_depth + 1
        for next_node in graph.successors(node):
            if next_node not in depth:
                depth[next_node] = next_depth
                to_process.append((next_node, operation))
            elif depth[next_node] == next_depth:
                graph.nodes[next_node]["root_cause_ops"][operation] += 1


def add_invariant_graph_features(
    graph, root_cause_index, manifestation_index, call_index, edge_type_indexes
):
    """Modify graph in-place with features not linked to manifestation.

    Adds attributes to nodes for features based on the global graph structure.
    Along the way,
    all of the nodes which are manifestation points are noted,
    and a collection of those nodes is returned.

    """
    betweenness, diameter = component_information(graph)
    nodes_by_tag = defaultdict(set)
    for node, data in graph.nodes(data=True):
        data["distance_manifestation"] = INFINITY_MULTIPLIER * diameter[node]
        data["betweenness"] = betweenness[node]
        data["root_cause_ops"] = Counter()
        data["distance_root_cause"] = INFINITY_MULTIPLIER * diameter[node]
        out_degree_by_type = Counter(
            t for _u, _v, d in graph.out_edges(node, data="type") for t in d
        )
        for edge_type, edge_type_index in edge_type_indexes.items():
            data[f"{edge_type}_out_degree"] = out_degree_by_type[edge_type_index]
        for tag in data["tag"]:
            nodes_by_tag[tag].add(node)

    parallel_root_cause_distances(graph, nodes_by_tag[root_cause_index])

    for _node, data in graph.nodes(data=True):
        resolve_root_cause_op(data, call_index)

    graph.graph["manifestation_nodes"] = sorted(nodes_by_tag[manifestation_index])


LLVM_INTRISICS_TO_REMOVE = [
    "llvm.dbg.",
    "llvm.lifetime.",
    "llvm.invariant.",
    "llvm.launder.",
    "llvm.strip.",
]


def remove_llvm_internal_functions(graph):
    # list since we will be mutating graph during the loop
    for node in list(graph):
        data = graph.nodes[node]
        function_name = data.get("function") or ""
        if any(function_name.startswith(prefix) for prefix in LLVM_INTRISICS_TO_REMOVE):
            next_node = one(graph.successors(node))
            inbound = [
                (u, next_node, d)
                for u, _v, d in graph.in_edges(node, data=True)
                if d["type"] != "def_use"
            ]
            graph.remove_node(node)
            graph.add_edges_from(inbound)


def prepare_indexes_for_training(indexes):
    """Prepare feature dictionaries for further training.

    Within each feature dictionary, looking up previously-unseen values
    will automatically assign the next index. This function assumes that
    the input dictionary is well-formed (i.e. that its values contain no
    gaps).

    """
    result = defaultdict(lambda: defaultdict(count().__next__))
    for k, v in indexes.items():
        inner = defaultdict(count(len(v)).__next__)
        inner.update(v)
        result[k] = inner
    return result


def preprocess_graph(
    graph: nx.MultiDiGraph,
    indexes: Indexes,
    *,
    source_dir: Optional[PurePath] = None,
    cwe: Optional[int] = None,
) -> nx.DiGraph:
    """Preprocess a single graph in-place."""
    remove_llvm_internal_functions(graph)
    relativize_file_names(graph, source_dir)
    if cwe is not None:
        translate_manifestation_point_labels(graph, POSITIVE_LABELS.get(cwe))
    graph = merge_edges(graph)
    translate_categorical_features(graph, indexes)
    add_invariant_graph_features(
        graph,
        manifestation_index=indexes["tag"][MANIFESTATION_POINT_TAG],
        call_index=indexes["operation"][CALL_OPERATION],
        root_cause_index=indexes["tag"][ROOT_CAUSE_TAG],
        edge_type_indexes=indexes["type"],
    )
    if source_dir is not None:
        graph.graph["source_dir"] = str(source_dir)
    return graph


@click.command()
@click.option(
    "--training-indexes",
    type=click.Path(dir_okay=False, writable=True),
    default="indexes.json",
    show_default=True,
    help="File where feature dictionaries are stored.",
)
@click.option(
    "--train/--no-train",
    default=True,
    show_default=True,
    help="Should new values be allowed?",
)
@click.option(
    "--source-dir", help="Absolute path that was --source-dir when LLAP was configured."
)
@click.option("--cwe", type=click.IntRange(0, None), help="CWE Being processed")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, allow_dash=True),
    default="-",
    help="Where output is written.",
)
@click.argument(
    "inputs", nargs=-1, type=click.Path(exists=True, dir_okay=False, allow_dash=True)
)
def main(training_indexes, train, source_dir, cwe, output, inputs):
    """Preprocess Program Dependence Graphs.

    This script reads PGDs from lines of JSON on the input files. It
    converts categorical features to indexes, computes graph-based
    features, and writes lines of JSON to stdout.

    The categorical feature dictionaries are read from the file
    specified by the --training-indexes option. If --train is enabled
    (which is is by default), previously-unseen values are added to the
    dictionary. If --no-train is specified, missing values are
    translated to ``None``. That will be encoded with all zeros in the
    one-hot representation.

    Specifying --source-dir will transform file names
    to be relative to the named directory if possible.
    Note that you must provide an absolute path here.
    The file system is not accessed with regards to this path,
    so it doesn't actually have to exist.

    """
    if source_dir is not None:
        source_dir = PurePath(source_dir)

    try:
        with open(training_indexes, "r") as f:
            indexes = json.load(f)
    except FileNotFoundError:
        indexes = {}
    if train:
        indexes = prepare_indexes_for_training(indexes)

    try:
        with smart_open(output, "wt") as output_file:
            for graph in read_graph_file(read_all(inputs)):
                graph = preprocess_graph(graph, indexes, source_dir=source_dir, cwe=cwe)
                print(json.dumps(nx.node_link_data(graph)), file=output_file)
    finally:
        if train:
            with click.open_file(training_indexes, "w", atomic=True) as f:
                json.dump(indexes, f, indent=2)
