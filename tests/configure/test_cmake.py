from unittest import mock

import networkx as nx
import pytest

from hector_ml.configure import cmake, ninja
from hector_ml.configure.base import SourceFile
from tests.configure.test_ninja import SIMPLE_DEP_GRAPH
from tests.test_graphs import assert_same_graph


def test_get_sources_missing_cmakelists(tmp_path):
    with pytest.raises(FileNotFoundError):
        cmake.get_sources(tmp_path / "build", tmp_path / "hector", "foo")


def test_get_sources(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "hector").mkdir()
    (tmp_path / "build/CMakeLists.txt").touch()

    with mock.patch("subprocess.run", autospec=True), mock.patch(
        "hector_ml.configure.ninja.get_sources",
        autospec=True,
        return_value=[SourceFile(tmp_path / "build/foo.c", ("-I/lib",))],
    ) as ninja_sources:
        result = cmake.get_sources(tmp_path / "build", tmp_path / "hector", "foo")
        assert ninja_sources.mock_calls == [
            mock.call(tmp_path / "hector", tmp_path / "hector", "foo")
        ]
    assert result == [SourceFile(tmp_path / "build/foo.c", ("-I/lib",))]


DEPS_OUTPUT = """foo.o: #deps 2, deps mtime 1610487204991221912 (VALID)
    foo.c
    foo.h

bar.o: #deps 2, deps mtime 1610487204993680696 (VALID)
    bar.c
    bar.h

main.o: #deps 3, deps mtime 1610487204999199130 (VALID)
    main.c
    foo.h
    bar.h

"""


def test_extend_dependency_graph():
    depgraph = SIMPLE_DEP_GRAPH.copy()

    expected = SIMPLE_DEP_GRAPH.copy()
    expected.add_node("foo.h", kind=ninja.NodeKind.FILE)
    expected.add_edge("foo.h", "build-1")
    expected.add_edge("foo.h", "build-3")
    expected.add_node("bar.h", kind=ninja.NodeKind.FILE)
    expected.add_edge("bar.h", "build-2")
    expected.add_edge("bar.h", "build-3")

    with mock.patch("subprocess.run", autospec=True), mock.patch(
        "hector_ml.configure.ninja.ninja", autospec=True, return_value=DEPS_OUTPUT
    ):
        cmake._extend_dependency_graph_with_headers(depgraph)

    assert_same_graph(expected, depgraph)


def test_infer_target():
    g = nx.DiGraph()
    g.add_node(0, label="foo.c", shape="box")
    g.add_node(1, label="C_COMPILER__foo")
    g.add_edge(0, 1)
    g.add_node(2, label="foo.o", shape="box")
    g.add_edge(1, 2)
    g.add_node(3, label="C_DYNAMIC_LIBRARY_LINKER__libfoo")
    g.add_edge(2, 3)
    g.add_node(4, label="libfoo.so", shape="box")
    g.add_edge(3, 4)
    g.add_node(5, label="bar.c", shape="box")
    g.add_node(6, label="C_COMPILER__bar")
    g.add_edge(5, 6)
    g.add_node(7, label="bar.o", shape="box")
    g.add_edge(6, 7)
    g.add_edge(7, 3)
    g.add_node(8, label="C_EXECUTABLE_LINKER__entry1")
    g.add_edge(2, 8)
    g.add_edge(7, 8)
    g.add_node(9, label="entry1", shape="box")
    g.add_edge(8, 9)
    g.add_node(10, label="entry1.c", shape="box")
    g.add_node(11, label="C_COMPILER__entry1")
    g.add_edge(10, 11)
    g.add_node(12, label="entry1.o", shape="box")
    g.add_edge(11, 12)
    g.add_edge(12, 8)
    g.add_node(13, label="entry2.c", shape="box")
    g.add_node(14, label="C_COMPILER__entry2")
    g.add_edge(13, 14)
    g.add_node(15, label="entry2.o", shape="box")
    g.add_edge(14, 15)
    g.add_node(16, label="C_EXECUTABLE_LINKER__entry2")
    g.add_edge(2, 16)
    g.add_edge(7, 16)
    g.add_edge(15, 16)
    g.add_node(17, label="entry2", shape="box")
    g.add_edge(16, 17)

    g = ninja._dependency_graph_from_raw(g, None)
    target = cmake._infer_target(
        g,
        [
            {"filename": "foo.c", "line_number": 17, "label": "second_free"},
            {"filename": "bar.c", "line_number": 21, "label": "second_free"},
            {"filename": "entry1.c", "line_number": 23, "label": "second_free"},
        ],
    )
    assert target == "entry1"
