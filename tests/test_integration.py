import json
import subprocess
from unittest import mock

import networkx as nx
import numpy as np
import pytest
from click.testing import CliRunner

from hector_ml.cli import main
from hector_ml.features import feature_set, node_features

from .test_graphs import SmallGraph


def dummy_feature_stats():
    fs = feature_set(node_features(SmallGraph.INDEXES), SmallGraph.INDEXES)
    return {
        "mean": np.zeros(fs.total_width, dtype=np.float64),
        "std": np.ones(fs.total_width, dtype=np.float64),
    }


@pytest.mark.parametrize("eager_dataset", [False, True])
def test_preprocess_train_stats_predict(eager_dataset):
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("raw_graphs.nljson", "w") as f:
            json.dump(nx.node_link_data(SmallGraph.raw()), f)
        with mock.patch.dict(
            "hector_ml.preprocess.POSITIVE_LABELS", {0: SmallGraph.POSITIVE_LABEL}
        ):
            result = runner.invoke(
                main,
                ["preprocess", "--cwe", "0", "raw_graphs.nljson"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        with open("processed_graphs.nljson", "w") as f:
            f.write(result.stdout)

        np.savez("feature_stats.npz", **dummy_feature_stats())

        result = runner.invoke(
            main,
            [
                "train",
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--eager-dataset" if eager_dataset else "--lazy-dataset",
                "0",
                "model",
                "processed_graphs.nljson",
                "processed_graphs.nljson",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        with open("foo.c", "w") as f:
            for i in range(1, 40):
                line = SmallGraph.SOURCE_LINES.get(i, "")
                print(line, file=f)

        result = runner.invoke(
            main,
            [
                "stats",
                "--device",
                "cpu",
                "--predictions-csv",
                "test.csv",
                "--dump",
                "test.npz",
                "--source-dir",
                ".",
                "--roc-file",
                "roc.png",
                "model",
                "processed_graphs.nljson",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        result = runner.invoke(
            main,
            ["predict", "--device", "cpu", "model", "processed_graphs.nljson"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0


def test_preprocess_hyperopt():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("raw_graphs.nljson", "w") as f:
            json.dump(nx.node_link_data(SmallGraph.raw()), f)
        with mock.patch.dict(
            "hector_ml.preprocess.POSITIVE_LABELS", {0: SmallGraph.POSITIVE_LABEL}
        ):
            result = runner.invoke(
                main,
                ["preprocess", "--cwe", "0", "raw_graphs.nljson"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        with open("processed_graphs.nljson", "w") as f:
            f.write(result.stdout)

        np.savez("feature_stats.npz", **dummy_feature_stats())

        result = runner.invoke(
            main,
            [
                "hyperopt",
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--n-random-starts",
                "3",
                "--n-calls",
                "3",
                "--eager-dataset",
                "0",
                "model",
                "processed_graphs.nljson",
                "processed_graphs.nljson",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # again, to test checkpoint & continue
        result = runner.invoke(
            main,
            [
                "hyperopt",
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--n-random-starts",
                "2",
                "--n-calls",
                "5",
                "--lazy-dataset",
                "0",
                "models",
                "processed_graphs.nljson",
                "processed_graphs.nljson",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        with open("foo.c", "w") as f:
            for i in range(1, 40):
                line = SmallGraph.SOURCE_LINES.get(i, "")
                print(line, file=f)

        result = runner.invoke(
            main,
            [
                "stats",
                "--device",
                "cpu",
                "--predictions-csv",
                "test.csv",
                "--dump",
                "test.npz",
                "--source-dir",
                ".",
                "--roc-file",
                "roc.png",
                "models/best",
                "processed_graphs.nljson",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0


def test_train_test_split():
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("raw.txt", "w") as f:
            for i in range(10):
                print(i, file=f)
        result = runner.invoke(
            main, ["train_test_split", "raw.txt", "train.txt", "test.txt"]
        )
        assert result.exit_code == 0

        with open("raw.txt") as f:
            raw_lines = set(f)
        with open("train.txt") as f:
            train_lines = set(f)
        with open("test.txt") as f:
            test_lines = set(f)

        assert not train_lines & test_lines
        assert train_lines | test_lines == raw_lines
        assert len(test_lines) * 10 == len(raw_lines)


@pytest.mark.parametrize("undirected", [False, True])
def test_visualize(undirected):
    runner = CliRunner()
    with runner.isolated_filesystem():
        graph = SmallGraph.raw()
        if undirected:
            graph = graph.to_undirected()
        with open("graph.json", "w") as f:
            json.dump(nx.node_link_data(graph), f)
        result = runner.invoke(
            main, ["visualize", "--output-file", "graph.dot", "graph.json"]
        )
        assert result.exit_code == 0

        try:
            subprocess.run(["dot", "graph.dot"], check=True)
        except FileNotFoundError:
            # graphviz is not installed
            pass
