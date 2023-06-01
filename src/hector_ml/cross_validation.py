from __future__ import annotations

import json

import click
import ignite
import numpy as np
import scipy.special
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from structure2vec.graph_collate import REDUCTIONS

from hector_ml.click_helpers import smart_open
from hector_ml.features import EDGE_FEATURES, feature_set, node_features
from hector_ml.graphs import GraphDataset, JSONGraphs
from hector_ml.model import Predictor
from hector_ml.train import TrainingData, training_loop


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--device", type=torch.device, default="cuda", help="Device on which to run."
)
@click.option(
    "--indexes",
    type=click.Path(dir_okay=False, exists=True),
    default="indexes.json",
    show_default=True,
    help="File where feature dictionaries are stored.",
)
@click.option(
    "--embedding-dimensions",
    type=click.IntRange(1, None),
    default=16,
    show_default=True,
    help="Dimensionality of graph embedding.",
)
@click.option(
    "--embedding-steps",
    type=click.IntRange(1, None),
    default=4,
    show_default=True,
    help="Iterations of embedding algorithm.",
)
@click.option(
    "--embedding-reduction",
    type=click.Choice(list(REDUCTIONS)),
    default="first",
    show_default=True,
    help="Reduction method to use at end of embedding.",
)
@click.option(
    "--recursive-depth",
    type=click.IntRange(1, None),
    default=2,
    show_default=True,
    help="Depth of embedding DNN.",
)
@click.option(
    "--classifier-dimensions",
    type=click.IntRange(1, None),
    default=16,
    show_default=True,
    help="Dimensionality of classifier DNN.",
)
@click.option(
    "--classifier-depth",
    type=click.IntRange(1, None),
    default=2,
    show_default=True,
    help="Depth of classifier DNN.",
)
@click.option(
    "--batch-size",
    metavar="INT",
    type=click.IntRange(1, None),
    default=50,
    show_default=True,
    help="Training batch size",
)
@click.option(
    "--epochs",
    metavar="INT",
    type=click.IntRange(1, None),
    default=50,
    show_default=True,
    help="Training epochs",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.001,
    show_default=True,
    help="Learning rate for Adam optimizer.",
)
@click.option(
    "--betas",
    type=(float, float),
    default=(0.9, 0.999),
    show_default=True,
    help="Gradient running average decays for Adam optimizer.",
)
@click.option(
    "--validation-fold",
    type=click.IntRange(0, None),
    default=0,
    help="Index (zero-based) of graph to leave out.",
)
@click.argument("cwe", type=click.IntRange(0, None))
@click.argument("output_dir", type=click.Path(file_okay=False, writable=True))
@click.argument("validation_graphs", type=click.Path(dir_okay=False, exists=True))
@TrainingData.apply_arguments
def train(
    device: torch.device,
    indexes: str,
    embedding_dimensions: int,
    embedding_steps: int,
    embedding_reduction: str,
    recursive_depth: int,
    classifier_dimensions: int,
    classifier_depth: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    betas: tuple[float, float],
    cwe: int,
    output_dir: str,
    validation_fold: int,
    validation_graphs: str,
    **kwargs,
):
    output_dir = f"{output_dir}-{validation_fold}"
    with click.open_file(indexes, "r") as f:
        indexes = json.load(f)

    node_feature_set = feature_set(node_features(indexes), indexes)
    edge_feature_set = feature_set(EDGE_FEATURES, indexes)

    predictor = Predictor.from_params(
        model_params=dict(
            node_features=node_feature_set.total_width,
            edge_features=edge_feature_set.total_width,
            n_classes=2,
            embedding_dimensions=embedding_dimensions,
            embedding_steps=embedding_steps,
            recursive_depth=recursive_depth,
            classifier_dimensions=classifier_dimensions,
            classifier_depth=classifier_depth,
            activation=torch.nn.PReLU,
        ),
        indexes=indexes,
        reduction_mode=embedding_reduction,
        cwe=cwe,
    )
    model = predictor.model.to(device=device)

    with smart_open(validation_graphs) as f:
        validation_graphs = list(JSONGraphs(f))
    validation_graph = validation_graphs.pop(validation_fold)
    extra_train_dataset = list(
        GraphDataset(
            validation_graphs, node_feature_set, edge_feature_set, predictor.depth_limit
        )
    )
    extra_test_dataset = list(
        GraphDataset(
            [validation_graph],
            node_feature_set,
            edge_feature_set,
            predictor.depth_limit,
        )
    )

    with TrainingData.from_parameters(
        node_feature_set, edge_feature_set, predictor.depth_limit, **kwargs
    ) as data:
        train_dataset = data.training + extra_train_dataset
        test_dataset = data.testing + extra_test_dataset

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, collate_fn=predictor.collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, collate_fn=predictor.collate_fn
        )

        history = training_loop(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            output_dir=output_dir,
            epochs=epochs,
            optimizer_args={"lr": learning_rate, "betas": betas},
            device=device,
        )

        predictor.save_aux(output_dir)

        final_test_loss = history["test"][-1]["loss"]
        print(f"Final test loss: {final_test_loss}")

    n_observations = len(extra_test_dataset)
    validation_loader = torch.utils.data.DataLoader(
        extra_test_dataset, batch_size=batch_size, collate_fn=predictor.collate_fn
    )
    logits = np.zeros(
        (n_observations, predictor.model_params["n_classes"]), dtype=np.float32
    )
    final_labels = np.zeros(n_observations, dtype=np.uint8)
    final_index = 0
    with torch.no_grad():
        for data, labels in validation_loader:
            this_batch_size = len(labels)
            logits[final_index : final_index + this_batch_size] = predictor.model(
                ignite.utils.convert_tensor(data, device=device)
            ).to(device="cpu")
            final_labels[final_index : final_index + this_batch_size] = labels
            final_index += this_batch_size
    np.savez(
        f"validation-fold-{validation_fold}.npz",
        logits=logits,
        labels=final_labels,
        cwe=cwe,
    )


@main.command()
@click.option("--roc-file", type=click.Path(dir_okay=False, writable=True))
@click.option("--output", type=click.Path(dir_okay=False, writable=True))
@click.argument("results", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def stats(roc_file: str | None, output: str | None, results: tuple[str, ...]):
    data = {}
    cwe = None
    for result in results:
        subdata = np.load(result)
        for k, v in subdata.items():
            if k == "cwe":
                if cwe is None:
                    cwe = v
                else:
                    assert cwe == v
            else:
                data.setdefault(k, []).append(v)
    data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
    logits = data["logits"]
    labels = data["labels"]

    probabilities = scipy.special.softmax(logits, axis=1)

    fpr, tpr, thresh = metrics.roc_curve(labels, probabilities[:, 1])
    decision_index = np.argmax(tpr - fpr)
    print("Decision threshold:", thresh[decision_index])
    print("True positive rate:", tpr[decision_index])
    print("False positive rate:", fpr[decision_index])
    predictions = (logits >= thresh[decision_index]).astype(logits.dtype)

    if output:
        np.savez(
            output,
            cwe=cwe,
            labels=labels,
            logits=logits,
            probabilities=probabilities,
            roc_fpr=fpr,
            roc_tpr=tpr,
            roc_thresh=thresh,
            decision_index=decision_index,
            predictions=predictions,
        )

    auc = metrics.auc(fpr, tpr)
    print("Area under ROC:", auc)
    if roc_file:
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"HECTOR - CWE{cwe} (AUC = {auc:.3})")
        plt.savefig(roc_file)

    cm = metrics.confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(cm)
