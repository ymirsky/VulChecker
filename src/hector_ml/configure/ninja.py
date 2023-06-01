"""Configure a source tree that is built with Ninja.

This assumes that build.ninja generation has already happened.
The "target" for this backend is used as a Ninja target,
which should build a single executable.

"""

import collections
import enum
import io
import itertools
import json
import pathlib
import shlex
import subprocess
from typing import Collection, Iterable, Mapping, NewType, Text

import networkx as nx

from hector_ml.configure.base import SourceFile

SOURCE_SUFFIXES = (".c", ".cpp", ".c++", ".cc", ".C", ".cxx")
FLAGS_TO_KEEP = ("-D", "-std=", "-m")


def ninja(*args, build_dir) -> Text:
    return subprocess.run(
        ["ninja", *args],
        stdout=subprocess.PIPE,
        cwd=build_dir,
        check=True,
        encoding="utf-8",
    ).stdout


class NodeKind(enum.Enum):
    BUILD = "ellipse"
    FILE = "box"


DependencyGraph = NewType("DependencyGraph", nx.DiGraph)


def load_dependency_graph(build_dir: pathlib.Path) -> DependencyGraph:
    raw_graph = nx.nx_pydot.read_dot(
        io.StringIO(ninja("-t", "graph", build_dir=build_dir))
    )
    return _dependency_graph_from_raw(raw_graph, build_dir)


def _clean_label(label):
    return label.strip('" ')


def _dependency_graph_from_raw(
    raw_graph: nx.DiGraph, build_dir: pathlib.Path
) -> DependencyGraph:
    default_node_shape = raw_graph.graph.get("node", {}).get("shape", "ellipse")

    node_translation = {}
    graph = nx.DiGraph()
    graph.graph["build_dir"] = build_dir
    build_counter = itertools.count()
    for node, data in raw_graph.nodes(data=True):
        kind = NodeKind(data.get("shape", default_node_shape))
        node_data = {"kind": kind}
        label = _clean_label(data.get("label", ""))
        if kind == NodeKind.BUILD:
            node_data["rule"] = label
            label = f"build-{next(build_counter)}"
        node_translation[node] = label
        graph.add_node(label, **node_data)
    for u, v, label in raw_graph.edges(data="label"):
        if label:
            build_node = f"build-{next(build_counter)}"
            graph.add_node(build_node, kind=NodeKind.BUILD, rule=_clean_label(label))
            graph.add_edge(node_translation[u], build_node)
            graph.add_edge(build_node, node_translation[v])
        else:
            graph.add_edge(node_translation[u], node_translation[v])
    return graph


def get_sources_from_dependency_graph(
    dependency_graph: DependencyGraph, build_dir: pathlib.Path, target: str
) -> Iterable[pathlib.Path]:
    if target not in dependency_graph:
        raise ValueError(f"Don't know about target {target!r}.")
    T = nx.bfs_tree(dependency_graph, target, reverse=True)
    for n in T:
        if dependency_graph.nodes[n]["kind"] == NodeKind.FILE and not T.out_degree(n):
            yield build_dir.joinpath(n).resolve()


def parse_compile_command(command_line, build_dir):
    for word in shlex.split(command_line):
        if word.startswith("-I"):
            path = build_dir.joinpath(word[2:])
            yield f"-I{path}"
        else:
            for prefix in FLAGS_TO_KEEP:
                if word.startswith(prefix):
                    yield word


def get_extra_flags(build_dir: pathlib.Path) -> Mapping[pathlib.Path, Collection[str]]:
    extra_flags = collections.defaultdict(list)
    rules = ninja("-t", "rules", build_dir=build_dir).splitlines()
    for rule in rules:
        comp_db = json.loads(ninja("-t", "compdb", rule, build_dir=build_dir))
        for command in comp_db:
            source_file = build_dir.joinpath(command["file"]).resolve()
            if source_file.suffix in SOURCE_SUFFIXES:
                extra_flags[source_file].extend(
                    parse_compile_command(command["command"], build_dir)
                )
    return {k: tuple(v) for k, v in extra_flags.items()}


def get_targets(build_dir: pathlib.Path, hector_dir: pathlib.Path) -> Iterable[str]:
    ninja_build_file = build_dir / "build.ninja"
    if not ninja_build_file.exists():
        raise FileNotFoundError(ninja_build_file)
    depgraph = load_dependency_graph(build_dir)
    for node, kind in depgraph.nodes(data="kind"):
        if kind == NodeKind.FILE and depgraph.in_degree(node):
            yield node


def get_sources(
    build_dir: pathlib.Path, hector_dir: pathlib.Path, target: str
) -> Iterable[SourceFile]:
    ninja_build_file = build_dir / "build.ninja"
    if not ninja_build_file.exists():
        raise FileNotFoundError(ninja_build_file)

    extra_flags = get_extra_flags(build_dir)

    for source_file_path in get_sources_from_dependency_graph(
        load_dependency_graph(build_dir), build_dir, target
    ):
        if source_file_path.suffix in SOURCE_SUFFIXES:
            yield SourceFile(source_file_path, extra_flags[source_file_path])


def get_reconfigure_inputs(
    build_dir: pathlib.Path, hector_dir: pathlib.Path, target: str
) -> Iterable[pathlib.Path]:
    yield build_dir / "build.ninja"
