import json
from typing import Optional

import click
import numpy as np
from structure2vec.graph_collate import graph_collate

from hector_ml.click_helpers import smart_open
from hector_ml.features import EDGE_FEATURES, FeatureKind, feature_set, node_features
from hector_ml.graphs import GraphDataset, JSONGraphs


@click.command()
@click.option(
    "--indexes", type=click.Path(exists=True, dir_okay=True), default="indexes.json"
)
@click.option(
    "--feature-stats",
    type=click.Path(writable=True, dir_okay=False),
    default="feature_stats.npz",
)
@click.option("--depth-limit", type=int)
@click.option("--epsilon", type=float, default=1e-6)
@click.argument("graphs")
def main(
    indexes: str,
    feature_stats: str,
    depth_limit: Optional[int],
    epsilon: float,
    graphs: str,
):
    with open(indexes) as f:
        indexes = json.load(f)
    node_feature_set = feature_set(node_features(indexes), indexes)
    edge_feature_set = feature_set(EDGE_FEATURES, indexes)
    with smart_open(graphs) as f:
        combined = graph_collate(
            list(
                GraphDataset(
                    JSONGraphs(f), node_feature_set, edge_feature_set, depth_limit
                )
            )
        )

    node_feature_mat = np.asarray(combined[0].features[0])

    means = np.zeros(node_feature_mat.shape[-1], dtype=node_feature_mat.dtype)
    stds = np.ones(node_feature_mat.shape[-1], dtype=node_feature_mat.dtype)
    col = 0
    for feature in node_feature_set.features:
        if feature.feature_kind == FeatureKind.numeric:
            feature_col = node_feature_mat[:, col]
            means[col] = np.mean(feature_col, dtype=np.float64)
            stds[col] = np.std(feature_col, dtype=np.float64)
        elif feature.feature_kind == FeatureKind.optional_numeric:
            feature_col = node_feature_mat[node_feature_mat[:, col + 1] == 0, col]
            means[col] = np.mean(feature_col, dtype=np.float64)
            stds[col] = np.std(feature_col, dtype=np.float64)
        col += feature.width

    stds[stds < epsilon] = epsilon

    np.savez(feature_stats, mean=means, std=stds)
