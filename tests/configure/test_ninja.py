import json
from pathlib import Path
from unittest import mock

import networkx as nx
import pytest

from hector_ml.configure import ninja
from hector_ml.configure.base import SourceFile
from tests.test_graphs import assert_same_graph

SIMPLE_DOT = """digraph ninja {
rankdir="LR"
node [fontsize=10, shape=box, height=0.25]
edge [fontsize=10]
"0x7faa3e604c60" [label="main"]
"0x7faa3e604b70" [label="C_EXECUTABLE_LINKER__main", shape=ellipse]
"0x7faa3e604b70" -> "0x7faa3e604c60"
"0x7faa3e604660" -> "0x7faa3e604b70" [arrowhead=none]
"0x7faa3e604800" -> "0x7faa3e604b70" [arrowhead=none]
"0x7faa3e6049e0" -> "0x7faa3e604b70" [arrowhead=none]
"0x7faa3e604660" [label="foo.o"]
"0x7faa3e6046e0" -> "0x7faa3e604660" [label=" C_COMPILER__foo"]
"0x7faa3e6046e0" [label="foo.c"]
"0x7faa3e604800" [label="bar.o"]
"0x7faa3e6048c0" -> "0x7faa3e604800" [label=" C_COMPILER__foo"]
"0x7faa3e6048c0" [label="bar.c"]
"0x7faa3e6049e0" [label="main.o"]
"0x7faa3e604a70" -> "0x7faa3e6049e0" [label=" C_COMPILER__foo"]
"0x7faa3e604a70" [label="main.c"]
}
"""

SIMPLE_DEP_GRAPH = nx.DiGraph(build_dir=None)
SIMPLE_DEP_GRAPH.add_node("foo.c", kind=ninja.NodeKind.FILE)
SIMPLE_DEP_GRAPH.add_node("foo.o", kind=ninja.NodeKind.FILE)
SIMPLE_DEP_GRAPH.add_node("bar.c", kind=ninja.NodeKind.FILE)
SIMPLE_DEP_GRAPH.add_node("bar.o", kind=ninja.NodeKind.FILE)
SIMPLE_DEP_GRAPH.add_node("main.c", kind=ninja.NodeKind.FILE)
SIMPLE_DEP_GRAPH.add_node("main.o", kind=ninja.NodeKind.FILE)
SIMPLE_DEP_GRAPH.add_node("main", kind=ninja.NodeKind.FILE)
SIMPLE_DEP_GRAPH.add_node(
    "build-0", kind=ninja.NodeKind.BUILD, rule="C_EXECUTABLE_LINKER__main"
)
SIMPLE_DEP_GRAPH.add_node("build-1", kind=ninja.NodeKind.BUILD, rule="C_COMPILER__foo")
SIMPLE_DEP_GRAPH.add_node("build-2", kind=ninja.NodeKind.BUILD, rule="C_COMPILER__foo")
SIMPLE_DEP_GRAPH.add_node("build-3", kind=ninja.NodeKind.BUILD, rule="C_COMPILER__foo")
SIMPLE_DEP_GRAPH.add_edge("build-0", "main")
SIMPLE_DEP_GRAPH.add_edge("foo.o", "build-0")
SIMPLE_DEP_GRAPH.add_edge("bar.o", "build-0")
SIMPLE_DEP_GRAPH.add_edge("main.o", "build-0")
SIMPLE_DEP_GRAPH.add_edge("build-1", "foo.o")
SIMPLE_DEP_GRAPH.add_edge("foo.c", "build-1")
SIMPLE_DEP_GRAPH.add_edge("build-2", "bar.o")
SIMPLE_DEP_GRAPH.add_edge("bar.c", "build-2")
SIMPLE_DEP_GRAPH.add_edge("build-3", "main.o")
SIMPLE_DEP_GRAPH.add_edge("main.c", "build-3")


def test_load_dependency_graph():
    with mock.patch(
        "hector_ml.configure.ninja.ninja", autospec=True, return_value=SIMPLE_DOT
    ):
        result = ninja.load_dependency_graph(None)

    assert_same_graph(SIMPLE_DEP_GRAPH, result)


def test_get_sources_from_dependency_graph_label_not_found():
    g = nx.DiGraph()

    with pytest.raises(ValueError):
        list(ninja.get_sources_from_dependency_graph(g, None, "foo"))


def test_get_sources_from_dependency_graph():
    assert set(
        ninja.get_sources_from_dependency_graph(
            SIMPLE_DEP_GRAPH, Path("/build"), "main"
        )
    ) == {Path("/build/foo.c"), Path("/build/bar.c"), Path("/build/main.c")}


def test_parse_compile_command():
    assert list(
        ninja.parse_compile_command("cc -O3 -Ilibfoo -I/opt/bar/libbar", Path("/build"))
    ) == ["-I/build/libfoo", "-I/opt/bar/libbar"]


def test_get_extra_flags():
    with mock.patch(
        "hector_ml.configure.ninja.ninja",
        side_effect=[
            "foo\n",
            json.dumps(
                [
                    {"file": "foo.c", "command": "cc -o foo.o -O3 -Ilibfoo -DFOO=bar"},
                    {"file": "foo.o", "command": "ll -o foo -llibfoo foo.o"},
                ]
            ),
        ],
        autospec=True,
    ):
        includes = ninja.get_extra_flags(Path("/build"))
    assert includes == {Path("/build/foo.c"): ("-I/build/libfoo", "-DFOO=bar")}


def test_get_sources_no_build_ninja(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(ninja.get_sources(tmp_path / "build", tmp_path / "hector", "foo"))


def test_get_sources(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build/build.ninja").touch()
    with mock.patch(
        "hector_ml.configure.ninja.get_extra_flags",
        autospec=True,
        return_value={
            tmp_path / "build/foo.c": ("-I/build",),
            tmp_path / "build/bar.c": ("-I/build",),
            tmp_path / "build/main.c": ("-I/build",),
        },
    ), mock.patch(
        "hector_ml.configure.ninja.load_dependency_graph",
        autospec=True,
        return_value=SIMPLE_DEP_GRAPH,
    ):
        sources = list(
            ninja.get_sources(tmp_path / "build", tmp_path / "hector", "foo.o")
        )
    assert sources == [SourceFile(tmp_path / "build/foo.c", ("-I/build",))]


def test_get_targets(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build/build.ninja").touch()
    with mock.patch(
        "hector_ml.configure.ninja.load_dependency_graph",
        autospec=True,
        return_value=SIMPLE_DEP_GRAPH,
    ):
        targets = set(ninja.get_targets(tmp_path / "build", tmp_path / "hector"))
    assert targets == {"foo.o", "bar.o", "main.o", "main"}
