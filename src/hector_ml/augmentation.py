from __future__ import annotations

import json
import logging
import random
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from typing import TypeVar

import attr
import click
import networkx as nx
from more_itertools import first, one

from hector_ml.click_helpers import smart_open
from hector_ml.preprocess import remove_llvm_internal_functions

T = TypeVar("T")

LABELS = "label"
TAGS = "tag"
ROOT_CAUSE = "root_cause"
MANIFESTATION_POINT = "manifestation"
EDGE_TYPE = "type"
CONTROL_FLOW = "control_flow"

LABEL_TO_TAG = {
    "stack_overflow": MANIFESTATION_POINT,
    "heap_overflow": MANIFESTATION_POINT,
    "overflowed_call": MANIFESTATION_POINT,
    "underflowed_call": MANIFESTATION_POINT,
    "second_free": MANIFESTATION_POINT,
    "use_after_free": MANIFESTATION_POINT,
    "declared_buffer": ROOT_CAUSE,
    "overflowed_variable": ROOT_CAUSE,
    "underflowed_variable": ROOT_CAUSE,
    "first_free": ROOT_CAUSE,
    "freed_variable": ROOT_CAUSE,
}

log = logging.getLogger(__name__)


@attr.frozen()
class PDGIndexes:
    tag_index: dict[str, set[int]]
    label_index: dict[str, set[int]]
    max_node: int

    @classmethod
    def from_graph(cls, graph: nx.MultiDiGraph[T]) -> PDGIndexes:
        tag_index = defaultdict(set)
        label_index = defaultdict(set)
        max_node = -1
        for n, data in graph.nodes(data=True):
            max_node = max(n, max_node)
            for tag in data[TAGS]:
                tag_index[tag].add(n)
            for label in data[LABELS]:
                label_index[LABEL_TO_TAG[label]].add(n)
        return cls(tag_index, label_index, max_node)


def path_subgraph(
    G: nx.Graph[T], start: T, end: T, max_length: int = None, min_length: int = None
) -> nx.Graph[T] | None:
    forward_nodes = {start} | {
        v for u, v in nx.bfs_edges(G, start, depth_limit=max_length)
    }
    reverse_nodes = {end} | {
        v for u, v in nx.bfs_edges(G, end, reverse=True, depth_limit=max_length)
    }
    nodes_to_keep = forward_nodes & reverse_nodes

    if not nodes_to_keep:
        return None

    SG = G.__class__()
    SG.graph.update(G.graph)

    translation = {}
    translation[start] = 0
    SG.add_node(0, **G.nodes[start])
    for i, n in enumerate(nodes_to_keep - {start, end}, start=1):
        translation[n] = i
        SG.add_node(i, **G.nodes[n])
    translation[end] = len(translation)
    SG.add_node(translation[end], **G.nodes[end])

    if SG.is_multigraph():
        SG.add_edges_from(
            (translation[n], translation[nbr], key, d)
            for n, nbrs in G.adj.items()
            if n in translation
            for nbr, keydict in nbrs.items()
            if nbr in translation
            for key, d in keydict.items()
        )
    else:
        SG.add_edges_from(
            (translation[n], translation[nbr], d)
            for n, nbrs in G.adj.items()
            if n in translation
            for nbr, d in nbrs.items()
            if nbr in translation
        )

    if (
        min_length is not None
        and nx.shortest_path_length(SG, translation[start], translation[end])
        < min_length
    ):
        return None

    return SG


def extract_malicious_path(
    graph: nx.MultiDiGraph, max_length: int = None, min_length: int = 3
) -> nx.MultiDiGraph | None:
    """Extracts benign paths from the graph."""
    indexes = PDGIndexes.from_graph(graph)

    true_manifestation_points = (
        indexes.label_index[MANIFESTATION_POINT]
        & indexes.tag_index[MANIFESTATION_POINT]
    )
    if true_manifestation_points:
        true_root_cause = first(
            indexes.label_index[ROOT_CAUSE] & indexes.tag_index[ROOT_CAUSE]
        )
        true_manifestation_point = first(true_manifestation_points)

        return path_subgraph(
            graph, true_root_cause, true_manifestation_point, max_length, min_length
        )
    else:
        return None


def extract_benign_paths(
    graph: nx.MultiDiGraph, max_length: int = None, min_length: int = 3
) -> Iterator[nx.MultiDiGraph]:
    indexes = PDGIndexes.from_graph(graph)
    root_causes = indexes.tag_index[ROOT_CAUSE]
    manifestations = indexes.tag_index[MANIFESTATION_POINT]
    for false_root_cause in root_causes:
        false_manifestation_point = None
        for u, v in nx.bfs_edges(graph, false_root_cause, depth_limit=max_length):
            if v in manifestations:
                false_manifestation_point = v
                break
        if false_manifestation_point is not None:
            SG = path_subgraph(
                graph,
                false_root_cause,
                false_manifestation_point,
                max_length,
                min_length,
            )
            if SG is not None:
                yield SG


def draw_random_path(
    graph: nx.MultiDiGraph, max_length: int, rng: random.Random
) -> list[int]:
    path = [rng.choice(list(graph))]
    for u, v in nx.dfs_edges(graph, path[0], depth_limit=max_length):
        if u != path[-1]:
            break
        path.append(v)
    return path


def cfg_view(pdg: nx.MultiDiGraph) -> nx.MultiDiGraph:
    def cfg_edge_filter(*e):
        return pdg.edges[e][EDGE_TYPE] == CONTROL_FLOW

    return nx.subgraph_view(pdg, filter_edge=cfg_edge_filter)


def splice_cfg(pdg: nx.MultiDiGraph, u, v, x, y):
    """Splice the control flow of a PDG.

    The edge u-v will be removed,
    and edges u-x and y-v will be created.

    """
    cfg = cfg_view(pdg)
    # HACK: Sometimes, we're getting parallel control flow edges,
    # even though that shouldn't happen.
    # Once this is fixed, revert this back to
    # key_to_remove = one(cfg.succ[u][v])
    # -gmacon3 2021-06-04
    key_to_remove = first(cfg.succ[u][v])
    pdg.remove_edge(u, v, key_to_remove)
    pdg.add_edge(u, x, type=CONTROL_FLOW, dtype="void")
    pdg.add_edge(y, v, type=CONTROL_FLOW, dtype="void")


def nodes_in_range(graph: nx.MultiGraph, start: list[int], depth_limit: int):
    depths = dict.fromkeys(start, 0)
    work = deque(start)
    while work:
        u = work.popleft()
        depth = depths[u]
        if depth >= depth_limit:
            continue
        for v in graph.neighbors(u):
            if v not in depths:
                depths[v] = depth + 1
                work.append(v)
    return depths.keys()


def augment(
    graph: nx.MultiDiGraph,
    juliet_types: Iterator[nx.MultiDiGraph],
    min_length: int,
    max_length: int,
    margin: int,
    attempts: int = 10,
    rng: random.Random = None,
):
    if rng is None:
        rng = random.Random()

    juliet_types = [iter(t) for t in juliet_types]

    unused_nodes = set(graph)
    next_node_id = max(graph) + 1
    unused_cfg = nx.subgraph_view(
        cfg_view(graph), filter_node=unused_nodes.__contains__
    )

    failures = 0
    while failures < attempts:
        next_path = draw_random_path(
            unused_cfg, rng.randint(min_length, max_length), rng
        )
        if len(next_path) < min_length:
            failures += 1
            continue
        failures = 0

        while True:
            if not juliet_types:
                return
            juliet_type_idx = rng.randrange(len(juliet_types))
            juliet = juliet_types[juliet_type_idx]
            try:
                juliet_pdg = next(juliet)
            except StopIteration:
                juliet_types.pop(juliet_type_idx)
                continue
            juliet_pdg = nx.convert_node_labels_to_integers(juliet_pdg, next_node_id)

            juliet_nodes = list(juliet_pdg)
            juliet_cfg = cfg_view(juliet_pdg)
            juliet_path = nx.shortest_path(
                juliet_cfg, juliet_nodes[0], juliet_nodes[-1]
            )
            mid_juliet = juliet_path[len(juliet_path) // 2 : len(juliet_path) // 2 + 2]
            next_node_id += len(juliet_pdg)
            break

        graph.update(juliet_pdg)
        splice_cfg(graph, next_path[0], next_path[1], juliet_nodes[0], mid_juliet[0])
        splice_cfg(graph, next_path[-2], next_path[-1], mid_juliet[1], juliet_nodes[-1])
        graph.remove_edge(
            mid_juliet[0],
            mid_juliet[1],
            one(juliet_cfg.succ[mid_juliet[0]][mid_juliet[1]]),
        )

        unused_nodes.difference_update(
            nodes_in_range(graph.to_undirected(as_view=True), next_path, margin)
        )


def load_juliet_fragments(
    lines: Iterable[str],
) -> tuple[list[nx.MultiDiGraph], list[nx.MultiDiGraph]]:
    good_fragments = []
    bad_fragments = []
    for i, line in enumerate(lines, start=1):
        try:
            juliet_pdg = nx.node_link_graph(json.loads(line), directed=True)
            remove_llvm_internal_functions(juliet_pdg)
            if "omitgood" in juliet_pdg.graph["file"]:
                fragment = extract_malicious_path(juliet_pdg)
                if fragment is not None:
                    bad_fragments.append(fragment)
            elif "omitbad" in juliet_pdg.graph["file"]:
                good_fragments.extend(extract_benign_paths(juliet_pdg))
        except Exception:
            log.exception("Error loading Juliet fragment line %d", i)
    return good_fragments, bad_fragments


@click.command()  # noqa: C901
@click.option("--seed", type=int, help="Random seed for reproducibility.")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, allow_dash=True),
    default="-",
    help="Location where selected paths will be written.",
)
@click.option(
    "--min-path-length",
    type=click.IntRange(3, None),
    default=10,
    help="Minimum path length to augment",
)
@click.option(
    "--max-path-length",
    type=click.IntRange(4, None),
    default=20,
    help="Maximum path length to augment",
)
@click.option(
    "--margin",
    type=int,
    default=30,
    help="Minimum graph distance between inserted paths",
)
@click.option(
    "--max-positive-injections",
    type=int,
    help=(
        "Maximum number of vulnerable examples to insert into a single graph "
        "(implies --inject-positive)."
    ),
)
@click.option(
    "--max-negative-injections",
    type=int,
    help=(
        "Maximum number of not-vulnerable examples to insert into a single graph "
        "(implies --inject-negative)."
    ),
)
@click.option("--inject-positive", is_flag=True, help="Inject vulnerable examples.")
@click.option("--inject-negative", is_flag=True, help="Inject not-vulnerable examples.")
@click.argument("juliet", type=click.Path(dir_okay=False, exists=True))
@click.argument("real_world", type=click.Path(dir_okay=False, exists=True))
@click.pass_context
def main(
    ctx: click.Context,
    seed: int | None,
    output: str,
    min_path_length: int,
    max_path_length: int,
    margin: int,
    max_positive_injections: int | None,
    max_negative_injections: int | None,
    inject_positive: bool,
    inject_negative: bool,
    juliet: str,
    real_world: str,
):
    """Augment a real-world program with Juliet vulnerabilities.

    For each PDG in REAL_WORLD,
    random control flow paths
    between --min-path-length and --max-path-length are chosen
    and a vulnerable path from JULIET is inserted
    into the control flow
    split into two parts at the beginning and end of the chosen path.
    This continues until there are no more JULIET examples
    or until there are no more suitable paths.

    You must specify at least one of --inject-positive or --inject-negative
    either directly or implicitly via --max-{positive,negative}-injections.

    Positive and negative examples are injected with equal probability
    until one set is exhausted (or the max for that type is reached).
    After that, the other type is injected unconditionally
    until it is exhausted (or its max is reached).

    """
    if max_path_length <= min_path_length:
        ctx.fail("Max path length must be strictly greater than min path length.")

    if max_positive_injections is not None:
        inject_positive = True
    if max_negative_injections is not None:
        inject_negative = True

    if not (inject_positive or inject_negative):
        ctx.fail(
            "You must specify at least one of --inject-positive or --inject-negative."
        )

    rng = random.Random(seed)

    with smart_open(juliet, "r") as juliet_file:
        good_fragments, bad_fragments = load_juliet_fragments(juliet_file)
    rng.shuffle(good_fragments)
    rng.shuffle(bad_fragments)

    fragment_types = []
    if inject_negative:
        if max_negative_injections is not None:
            good_fragments = good_fragments[:max_negative_injections]
        fragment_types.append(good_fragments)
    if inject_positive:
        if max_positive_injections is not None:
            bad_fragments = bad_fragments[:max_positive_injections]
        fragment_types.append(bad_fragments)
    with smart_open(output, "wt") as outfile, smart_open(
        real_world, "r"
    ) as real_world_file:
        for i, line in enumerate(real_world_file, start=1):
            graph = nx.node_link_graph(json.loads(line), directed=True)
            remove_llvm_internal_functions(graph)
            try:
                augment(
                    graph,
                    fragment_types,
                    min_path_length,
                    max_path_length,
                    margin,
                    rng=rng,
                )
            except Exception:
                log.exception("Failed to augment graph on line %d", i)
            else:
                print(json.dumps(nx.node_link_data(graph)), file=outfile)
