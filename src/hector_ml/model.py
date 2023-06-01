import enum
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, Union

import attr
import networkx as nx
import numpy as np
import torch
from structure2vec.discriminative_embedding import MeanFieldInference
from structure2vec.graph_collate import graph_collate

from hector_ml.compat import cached_property
from hector_ml.features import EDGE_FEATURES, feature_set, node_features
from hector_ml.graphs import mean_field_from_graph


def dnn(in_features, out_features, *, layers=2, activation=torch.nn.ReLU):
    steps = []
    in_shape = in_features
    for _ in range(layers):
        steps.extend((torch.nn.Linear(in_shape, out_features), activation()))
        in_shape = out_features
    return torch.nn.Sequential(*steps)


class Model(torch.nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        n_classes,
        *,
        embedding=MeanFieldInference,
        embedding_dimensions=16,
        embedding_steps=4,
        recursive_depth=2,
        classifier_dimensions=16,
        classifier_depth=2,
        activation=torch.nn.ReLU,
        mean=None,
        std=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mean is None:
            mean = torch.zeros(node_features, dtype=torch.float32)
        else:
            mean = torch.as_tensor(mean).clone().detach()
        self.register_buffer("mean", mean)
        if std is None:
            std = torch.ones(node_features, dtype=torch.float32)
        else:
            std = torch.as_tensor(std).clone().detach()
        self.register_buffer("std", std)
        recursive = dnn(
            embedding_dimensions,
            embedding_dimensions,
            layers=recursive_depth,
            activation=activation,
        )
        self.embedding = embedding(
            node_features,
            edge_features,
            embedding_dimensions,
            steps=embedding_steps,
            recursive=recursive,
        )
        self.batchnorm = torch.nn.BatchNorm1d(embedding_dimensions)
        self.classifier = dnn(
            embedding_dimensions,
            classifier_dimensions,
            layers=classifier_depth,
            activation=activation,
        )
        self.logits = torch.nn.Linear(
            classifier_dimensions, n_classes, dtype=torch.float64
        )

    def forward(self, x):
        x.features[0] -= self.mean
        x.features[0] /= self.std
        x = self.embedding(x)
        x = self.classifier(x).to(torch.float64)
        x = self.logits(x)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x


def _identity(x):
    return x


class PredictionStyle(enum.Enum):
    # Wrap values in a tuple to prevent these from turning into methods
    log_probabilities = (_identity,)
    probabilities = (np.exp,)
    classes = (partial(np.argmax, axis=-1),)


@attr.s(auto_attribs=True, frozen=True)
class Predictor:
    model: Model
    model_params: Mapping[str, Any]
    indexes: Mapping[str, Mapping[str, int]]
    reduction_mode: str
    cwe: int
    batch_size: int = 64

    @cached_property
    def collate_fn(self):
        return partial(graph_collate, mode=self.reduction_mode)

    @cached_property
    def depth_limit(self):
        return self.model_params["embedding_steps"]

    @cached_property
    def node_features(self):
        return feature_set(node_features(self.indexes), self.indexes)

    @cached_property
    def edge_features(self):
        return feature_set(EDGE_FEATURES, self.indexes)

    def _predict(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        graph_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, collate_fn=self.collate_fn
        )
        logits = np.zeros(
            (len(dataset), self.model_params["n_classes"]), dtype=np.float64
        )
        index = 0
        with torch.no_grad():
            for batch in graph_loader:
                batch_size = batch.structure[0].shape[0]
                logits[index : index + batch_size] = self.model(batch)
                index += batch_size
        return logits

    def predict_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        style: PredictionStyle = PredictionStyle.log_probabilities,
    ) -> np.ndarray:
        logits = self._predict(dataset)
        return style.value[0](logits)

    def predict_graphs(
        self,
        graphs: Iterable[nx.Graph],
        style: PredictionStyle = PredictionStyle.log_probabilities,
    ) -> np.ndarray:
        dataset = [
            mean_field_from_graph(g, self.node_features, self.edge_features)
            for g in graphs
        ]
        return self.predict_dataset(dataset, style)

    def save_aux(self, dir_path: Union[bytes, str, os.PathLike]):
        dir_path = Path(dir_path)
        with open(dir_path / "aux.pkl", "wb") as f:
            pickle.dump(
                {
                    "model_params": self.model_params,
                    "indexes": self.indexes,
                    "reduction_mode": self.reduction_mode,
                    "cwe": self.cwe,
                },
                f,
            )

    @classmethod
    def from_params(
        cls,
        model_params: Mapping[str, Any],
        indexes: Mapping[str, Mapping[str, int]],
        reduction_mode: str,
        cwe: int,
        batch_size: int = 64,
    ):
        model = Model(**model_params)
        return cls(
            model=model,
            model_params=model_params,
            indexes=indexes,
            reduction_mode=reduction_mode,
            cwe=cwe,
            batch_size=batch_size,
        )

    @classmethod
    def from_checkpoint_dir(
        cls,
        dir_path: Union[bytes, str, os.PathLike],
        batch_size: int = 64,
        map_location=None,
        pickle_module=pickle,
        pickle_load_args=None,
    ) -> "Predictor":
        dir_path = Path(dir_path)
        if pickle_load_args is None:
            pickle_load_args = {}

        aux_path = dir_path / "aux.pkl"
        with open(aux_path, "rb") as f:
            additional_data = pickle.load(f)
        model = Model(**additional_data["model_params"])

        model_state_path = next(iter(dir_path.glob("model_*")))
        model_state = torch.load(
            str(model_state_path),
            map_location=map_location,
            pickle_module=pickle_module,
            **pickle_load_args,
        )
        model.load_state_dict(model_state)

        return cls(
            model=model,
            model_params=additional_data["model_params"],
            indexes=additional_data["indexes"],
            reduction_mode=additional_data["reduction_mode"],
            cwe=additional_data["cwe"],
            batch_size=batch_size,
        )
