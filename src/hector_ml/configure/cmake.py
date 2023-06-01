import collections
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from more_itertools import one

from hector_ml.configure import ninja
from hector_ml.configure.base import SourceFile
from hector_ml.paths import relative_to_with_parents


def _cmake_to_ninja(build_dir: Path, hector_dir: Path):
    cmakelists_file = build_dir / "CMakeLists.txt"
    if not cmakelists_file.exists():
        raise FileNotFoundError(cmakelists_file)

    ninja_file = hector_dir / "build.ninja"
    if not ninja_file.exists():
        subprocess.run(
            ["cmake", "-GNinja", str(relative_to_with_parents(build_dir, hector_dir))],
            cwd=hector_dir,
            check=True,
        )
    else:
        subprocess.run(["ninja", "build.ninja"], cwd=hector_dir, check=True)
    subprocess.run(["ninja"], cwd=hector_dir)


def get_targets(build_dir: Path, hector_dir: Path) -> Iterable[str]:
    _cmake_to_ninja(build_dir, hector_dir)
    dependency_graph = ninja.load_dependency_graph(hector_dir)
    for node, data in dependency_graph.nodes(data=True):
        if (
            data["kind"] == ninja.NodeKind.BUILD
            and "_EXECUTABLE_LINKER_" in data["rule"]
        ):
            yield one(dependency_graph.successors(node))


def _extend_dependency_graph_with_headers(graph: ninja.DependencyGraph):
    build_dir = graph.graph["build_dir"]
    build = None
    for line in ninja.ninja("-t", "deps", build_dir=build_dir).splitlines():
        if build is None:
            target = line.partition(":")[0]
            build = one(graph.predecessors(target))
        else:
            dep = line.strip()
            if not dep:
                build = None
            else:
                graph.add_node(dep, kind=ninja.NodeKind.FILE)
                graph.add_edge(dep, build)


def _infer_target(
    dependency_graph: ninja.DependencyGraph, labels, prefix: Path = Path()
) -> Optional[str]:
    scores = collections.Counter(str(prefix / label["filename"]) for label in labels)
    processed = set(scores)
    work = collections.deque(scores)
    targets = set()
    while work:
        current_node = work.popleft()
        current_score = scores[current_node]
        try:
            node_data = dependency_graph.nodes[current_node]
        except KeyError:
            pass
        else:
            next_is_target = "_EXECUTABLE_LINKER_" in (node_data.get("rule") or "")
            for next_node in dependency_graph.successors(current_node):
                scores[next_node] += current_score
                if next_node not in processed:
                    if next_is_target:
                        targets.add(next_node)
                    work.append(next_node)
                    processed.add(next_node)
    for node, count in scores.most_common():
        if node in targets:
            return node


def infer_target(
    build_dir: Path, hector_dir: Path, *, labels_file_name: str = "hector_labels.json"
) -> Optional[str]:
    """Choose a target that covers the most labeled vulnerabilities.

    If there are multiple vulnerabilities that appear in only one target,
    and those targets are different,
    there's no way to pick a single target that covers all of the vulnerabilities.
    This should do the best possible.

    """
    _cmake_to_ninja(build_dir, hector_dir)
    dependency_graph = ninja.load_dependency_graph(hector_dir)
    _extend_dependency_graph_with_headers(dependency_graph)

    with open(build_dir / labels_file_name) as f:
        labels = json.load(f)

    return _infer_target(
        dependency_graph, labels, relative_to_with_parents(build_dir, hector_dir)
    )


def get_sources(
    build_dir: os.PathLike, hector_dir: os.PathLike, target: str
) -> Iterable[SourceFile]:
    _cmake_to_ninja(build_dir, hector_dir)
    return ninja.get_sources(hector_dir, hector_dir, target)


def get_reconfigure_inputs(
    build_dir: os.PathLike, hector_dir: os.PathLike, target: str
) -> Iterable[Path]:
    return Path(build_dir).glob("**/CMakeLists.txt")
