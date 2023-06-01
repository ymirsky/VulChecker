import networkx as nx
import numpy as np
import torch
from structure2vec.graph_collate import Graph, graph_collate

from hector_ml.model import Model, Predictor

node_features = 4
edge_features = 5
n_classes = 2


def dummy_data(model):
    graphs = [nx.fast_gnp_random_graph(n, 0.25) for n in (10, 11, 12)]
    node_feature_mat = [
        np.random.normal(size=(len(g.nodes), node_features)).astype(np.float32)
        for g in graphs
    ]
    edge_feature_mat = [
        np.random.normal(size=(len(g.edges), edge_features)).astype(np.float32)
        for g in graphs
    ]
    feature_mats = list(zip(node_feature_mat, edge_feature_mat))
    return [
        Graph(model.embedding.graph_structure(g), f)
        for g, f in zip(graphs, feature_mats)
    ]


def test_model_forward():
    model = Model(
        node_features=node_features, edge_features=edge_features, n_classes=n_classes
    )
    data = dummy_data(model)
    with torch.no_grad():
        logits = model(graph_collate(data))
    assert logits.shape == (len(data), n_classes)


def test_save_load_model(tmp_path):
    model_params = {
        "node_features": node_features,
        "edge_features": edge_features,
        "n_classes": n_classes,
    }
    predictor = Predictor.from_params(
        model_params=model_params, indexes={}, reduction_mode="sum", cwe=0
    )
    model = predictor.model
    torch.save(model.state_dict(), tmp_path / "model_model_0.pt")
    predictor.save_aux(tmp_path)
    predictor = Predictor.from_checkpoint_dir(tmp_path)

    data = dummy_data(model)
    with torch.no_grad():
        orig_logits = model(graph_collate(data))
        new_logits = predictor.predict_dataset(data)
    assert np.all(orig_logits.numpy() == new_logits)
