import json
import os
import pickle
from contextlib import ExitStack
from pathlib import Path
from typing import ClassVar, Optional, Union

import attr
import click
import ignite
import numpy as np
import torch
from ignite.engine import Events
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from sklearn import metrics
from structure2vec.graph_collate import REDUCTIONS

from hector_ml.click_helpers import smart_open
from hector_ml.features import EDGE_FEATURES, FeatureSet, feature_set, node_features
from hector_ml.graphs import GraphDataset, JSONGraphs
from hector_ml.model import Predictor


class AUC(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._preds = []
        self._labels = []
        super(AUC, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._preds = []
        self._labels = []
        super(AUC, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach().cpu().numpy(), output[1].detach().cpu().numpy()

        self._preds.append(y_pred)
        self._labels.append(y)

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "Must have at least one example before it can be computed."
            )
        P = np.concatenate(self._preds)
        Y = np.concatenate(self._labels)
        # fpr, tpr, thresholds = metrics.roc_curve(Y, P[:,1])
        return metrics.roc_auc_score(Y, P[:, 1], max_fpr=0.5)
        # return metrics.auc(fpr, tpr)


class CollectHistory:
    def __init__(self):
        self.history = []

    def __call__(self, engine):
        # Some of the metrics are tensors, and others are floats.
        # np.asarray handles both types.
        # However, np.asarray(some_tensor) returns a view of the tensor,
        # so call tolist to get an independent copy.
        self.history.append(
            {k: np.asarray(v).tolist() for k, v in engine.state.metrics.items()}
        )
        engine.logger.info(
            "Epoch[%d] metrics: %r", engine.state.epoch, engine.state.metrics
        )


def activated_output_transform(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y


def build_evaluator(title, model, loss, device):
    evaluator = ignite.engine.create_supervised_evaluator(
        model,
        metrics={
            "loss": ignite.metrics.Loss(loss),
            "auc": AUC(),
            "precision": ignite.metrics.Precision(),
            "recall": ignite.metrics.Recall(),
        },
        device=device,
    )
    evaluator.logger = ignite.utils.setup_logger(title)
    history_collector = CollectHistory()
    evaluator.add_event_handler(Events.COMPLETED, history_collector)
    return evaluator, history_collector


def training_loop(
    *,
    model,
    train_loader,
    test_loader,
    output_dir,
    epochs,
    patience,
    optimizer_args=None,
    keep_best=False,
    device,
):
    if optimizer_args is None:
        optimizer_args = {}
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)
    loss = torch.nn.NLLLoss()

    trainer = ignite.engine.create_supervised_trainer(
        model, optimizer, loss, device=device
    )
    trainer.logger = ignite.utils.setup_logger("trainer")
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, ignite.handlers.TerminateOnNan()
    )

    train_eval, train_history = build_evaluator("train_eval", model, loss, device)
    test_eval, test_history = build_evaluator("test_eval", model, loss, device)
    test_eval.add_event_handler(
        Events.COMPLETED,
        ignite.handlers.EarlyStopping(
            patience=patience,
            score_function=lambda e: e.state.metrics["auc"],
            trainer=trainer,
            min_delta=1e-6,
        ),
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(trainer):
        train_eval.run(train_loader)
        test_eval.run(test_loader)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ignite.handlers.ModelCheckpoint(
            output_dir,
            "model",
            score_function=(
                (lambda e: test_eval.state.metrics["auc"]) if keep_best else None
            ),
            require_empty=False,
        ),
        {"model": model},
    )

    trainer.run(train_loader, max_epochs=epochs)

    return {"train": train_history.history, "test": test_history.history}


@attr.define()
class TrainingData:
    training: GraphDataset
    testing: GraphDataset
    _stack: ExitStack

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    _ARGUMENTS: ClassVar = [
        click.option(
            "--eager-dataset/--lazy-dataset",
            default=True,
            show_default=True,
            help="Load entire dataset into memory in advance.",
        ),
        click.argument("training_graphs", type=click.Path(dir_okay=False, exists=True)),
        click.argument("testing_graphs", type=click.Path(dir_okay=False, exists=True)),
    ]

    @classmethod
    def apply_arguments(cls, f):
        for applier in reversed(cls._ARGUMENTS):
            f = applier(f)
        return f

    @classmethod
    def from_parameters(
        cls,
        node_feature_set: FeatureSet,
        edge_feature_set: FeatureSet,
        depth_limit: Optional[int],
        *,
        eager_dataset: bool,
        training_graphs: Union[str, os.PathLike],
        testing_graphs: Union[str, os.PathLike],
    ) -> "TrainingData":
        with ExitStack() as stack:
            train_f = stack.enter_context(smart_open(training_graphs, "rt"))
            train_dataset = GraphDataset(
                JSONGraphs(train_f), node_feature_set, edge_feature_set, depth_limit
            )
            test_f = stack.enter_context(smart_open(testing_graphs, "rt"))
            test_dataset = GraphDataset(
                JSONGraphs(test_f), node_feature_set, edge_feature_set, depth_limit
            )

            if eager_dataset:
                train_cache_path = training_graphs + ".cache.pkl"
                test_cache_path = testing_graphs + ".cache.pkl"
                if os.path.isfile(train_cache_path):
                    print("Loading train from cache")
                    with open(train_cache_path, "rb") as handle:
                        train_dataset = pickle.load(handle)
                else:
                    print("Loading train")
                    train_dataset = list(train_dataset)
                    print("Saving train as cache")
                    with open(train_cache_path, "wb") as handle:
                        pickle.dump(
                            train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL
                        )
                if os.path.isfile(test_cache_path):
                    print("Loading test from cache")
                    with open(test_cache_path, "rb") as handle:
                        test_dataset = pickle.load(handle)
                else:
                    print("Loading test")
                    test_dataset = list(test_dataset)
                    print("Saving test as cache")
                    with open(test_cache_path, "wb") as handle:
                        pickle.dump(
                            test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL
                        )
                stack_out = ExitStack()
            else:
                stack_out = stack.pop_all()

            return cls(training=train_dataset, testing=test_dataset, stack=stack_out)


@click.command()
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
    "--feature-stats",
    type=click.Path(dir_okay=False, exists=True),
    default="feature_stats.npz",
    show_default=True,
    help="File where feature statistics are stored.",
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
    "--patience",
    metavar="INT",
    type=click.IntRange(1, None),
    default=10,
    show_default=True,
    help="Earlystopping Patience",
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
@click.option("--fine-tune", is_flag=True, help="Fine-tune an existing model.")
@click.option(
    "--existing",
    type=click.Path(file_okay=False),
    help="Model path to load (default: same as output).",
)
@click.option(
    "--keep-best", is_flag=True, help="Keep the best model instead of the last one."
)
@click.argument("cwe", type=click.IntRange(0, None))
@click.argument("output_dir", type=click.Path(file_okay=False, writable=True))
@TrainingData.apply_arguments
@click.pass_context
def main(
    ctx: click.Context,
    device,
    indexes,
    feature_stats,
    embedding_dimensions,
    embedding_steps,
    embedding_reduction,
    recursive_depth,
    classifier_dimensions,
    classifier_depth,
    batch_size,
    epochs,
    patience,
    learning_rate,
    betas,
    fine_tune,
    existing,
    keep_best,
    cwe,
    output_dir,
    **kwargs,
):
    if existing is not None:
        fine_tune = True
    if fine_tune and existing is None:
        existing = output_dir

    if existing is not None:
        predictor = Predictor.from_checkpoint_dir(existing, map_location=device)
    else:
        with click.open_file(indexes, "r") as f:
            indexes = json.load(f)

        node_feature_set = feature_set(node_features(indexes), indexes)
        edge_feature_set = feature_set(EDGE_FEATURES, indexes)

        feature_stats = np.load(feature_stats)

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
                mean=feature_stats["mean"],
                std=feature_stats["std"],
            ),
            indexes=indexes,
            reduction_mode=embedding_reduction,
            cwe=cwe,
        )

        output_path = Path(output_dir)
        if output_path.exists() and next(output_path.iterdir(), None):
            click.echo(
                "Output path exists and is not empty, refusing to overwrite it.",
                err=True,
            )
            ctx.exit(1)
        output_path.mkdir(parents=True, exist_ok=True)
    model = predictor.model.to(device=device)

    predictor.save_aux(output_dir)

    with TrainingData.from_parameters(
        predictor.node_features,
        predictor.edge_features,
        predictor.depth_limit,
        **kwargs,
    ) as data:
        train_loader = torch.utils.data.DataLoader(
            data.training, batch_size=batch_size, collate_fn=predictor.collate_fn
        )
        test_loader = torch.utils.data.DataLoader(
            data.testing, batch_size=batch_size, collate_fn=predictor.collate_fn
        )

        history = training_loop(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            output_dir=output_dir,
            epochs=epochs,
            patience=patience,
            optimizer_args={"lr": learning_rate, "betas": betas},
            keep_best=keep_best,
            device=device,
        )

        final_test_loss = history["test"][-1]["loss"]
        print(f"Final test loss: {final_test_loss}")
