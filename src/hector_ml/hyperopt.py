#!/usr/bin/env python3

import json
import os.path
import pathlib
from functools import partial
from itertools import count

import click
import numpy as np
import skopt
import torch
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver, DeltaYStopper
from skopt.space import Categorical, Integer, Real
from structure2vec.graph_collate import REDUCTIONS

from hector_ml.features import EDGE_FEATURES, feature_set, node_features
from hector_ml.model import Predictor
from hector_ml.train import TrainingData, training_loop


def point_to_options(dimensions, point):
    result = []
    for dim, val in zip(dimensions, point):
        result.append("--{}".format(dim.name.replace("_", "-")))
        result.append(str(val))
    return result


def point_to_str(point):
    return "-".join(str(x) for x in point)


def symlink_force(target, link):
    target = pathlib.Path(target)
    link = pathlib.Path(link)
    if not target.is_absolute():
        try:
            target = target.relative_to(link.parent)
        except ValueError:
            target = target.resolve()
    try:
        link.symlink_to(target)
    except FileExistsError:
        link.unlink()
        link.symlink_to(target)


def progress(output_dir, result):
    dimensions = result.space.dimensions
    print("Best loss so far:", result.fun)
    print("Best options so far:", " ".join(point_to_options(dimensions, result.x)))
    best_dir = "model-{}-{}".format(
        result.x_iters.index(result.x), point_to_str(result.x)
    )
    print("Best model in directory:", best_dir)
    symlink_force(os.path.join(output_dir, best_dir), os.path.join(output_dir, "best"))


@click.command()  # noqa: C901: function too complex
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
    default=32,
    show_default=True,
    help="Maximum dimensionality of graph embedding.",
)
@click.option(
    "--embedding-steps",
    type=click.IntRange(1, None),
    default=16,
    show_default=True,
    help="Maximum iterations of embedding algorithm.",
)
@click.option(
    "--recursive-depth",
    type=click.IntRange(1, None),
    default=6,
    show_default=True,
    help="Maximum depth of embedding DNN.",
)
@click.option(
    "--embedding-reductions",
    default="first",
    show_default=True,
    help=(
        "Comma-separated values to choose from for the reduction. "
        "Select at least one of first, mean, sum."
    ),
)
@click.option(
    "--classifier-dimensions",
    type=click.IntRange(1, None),
    default=32,
    show_default=True,
    help="Maximum dimensionality of classifier DNN.",
)
@click.option(
    "--classifier-depth",
    type=click.IntRange(1, None),
    default=6,
    show_default=True,
    help="Maximum depth of classifier DNN.",
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
    help="Maximum training epochs",
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
    "--n-jobs",
    type=int,
    envvar="SLURM_JOB_CPUS_PER_NODE",
    default=-1,
    show_default=True,
    help="Concurrent jobs to use while finding next hyperparameters.",
)
@click.option(
    "--n-calls",
    type=click.IntRange(1, None),
    default=100,
    show_default=True,
    help="Number of training runs to use.",
)
@click.option(
    "--n-random-starts",
    type=click.IntRange(1, None),
    default=10,
    show_default=True,
    help="Number of random points to seed optimization.",
)
@click.argument("cwe", type=click.IntRange(0, None))
@click.argument("output_dir", type=click.Path(file_okay=False, writable=True))
@TrainingData.apply_arguments
@click.pass_context
def main(
    ctx,
    device,
    indexes,
    feature_stats,
    embedding_dimensions,
    embedding_steps,
    recursive_depth,
    embedding_reductions,
    classifier_dimensions,
    classifier_depth,
    batch_size,
    epochs,
    patience,
    cwe,
    output_dir,
    n_jobs,
    n_calls,
    n_random_starts,
    **kwargs,
):
    """Optimize hyperparameters."""

    if not embedding_reductions:
        ctx.fail("You must provide at least one embedding reduction.")
    embedding_reductions = embedding_reductions.split(",")
    unknown_reductions = set(embedding_reductions) - REDUCTIONS.keys()
    if unknown_reductions:
        ctx.fail(f"Unknown embedding reductions: {unknown_reductions}")

    with click.open_file(indexes, "r") as f:
        indexes = json.load(f)

    feature_stats = np.load(feature_stats)

    node_feature_set = feature_set(node_features(indexes), indexes)
    edge_feature_set = feature_set(EDGE_FEATURES, indexes)

    dimensions = [
        Real(1e-6, 0.001, prior="log-uniform", name="adam.lr"),
        Real(0.85, 0.9, name="adam.betas.0"),
        Real(0.85, 0.999, name="adam.betas.1"),
        Integer(
            2,
            embedding_dimensions,
            prior="log-uniform",
            name="model.embedding_dimensions",
        ),
        Integer(5, embedding_steps, name="model.embedding_steps"),
        Integer(2, recursive_depth, name="model.recursive_depth"),
        Integer(
            2,
            classifier_dimensions,
            prior="log-uniform",
            name="model.classifier_dimensions",
        ),
        Integer(2, classifier_depth, name="model.classifier_depth"),
    ]

    if len(embedding_reductions) > 1:
        dimensions.append(Categorical(embedding_reductions, name="embedding_reduction"))

    os.makedirs(output_dir, exist_ok=True)

    def objective(x):
        params = {
            "model": {
                "node_features": node_feature_set.total_width,
                "edge_features": edge_feature_set.total_width,
                "n_classes": 2,
                "activation": torch.nn.PReLU,
                "mean": feature_stats["mean"],
                "std": feature_stats["std"],
            }
        }
        for d, v in zip(dimensions, x):
            name_parts = d.name.split(".")
            dest = params
            for part in name_parts[:-1]:
                dest = dest.setdefault(part, {})
            dest[name_parts[-1]] = v
        params["adam"]["betas"] = tuple(
            params["adam"]["betas"][str(i)] for i in range(len(params["adam"]["betas"]))
        )

        embedding_reduction = params.get("embedding_reduction", embedding_reductions[0])
        predictor = Predictor.from_params(
            model_params=params["model"],
            indexes=indexes,
            reduction_mode=embedding_reduction,
            cwe=cwe,
        )
        model = predictor.model.to(device)
        checkpoint_dir = os.path.join(
            output_dir, "model-{}-{}".format(next(objective_sequence), point_to_str(x))
        )

        with TrainingData.from_parameters(
            node_feature_set, edge_feature_set, predictor.depth_limit, **kwargs
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
                output_dir=checkpoint_dir,
                epochs=epochs,
                device=device,
                optimizer_args=params["adam"],
                patience=patience,
            )

        predictor.save_aux(checkpoint_dir)

        try:
            score = history["test"][-1]["auc"]
        except IndexError:
            score = 0

        # The not-less-than construction (instead of greater-than-or-equal)
        # gives us the correct behavior if score happnes to be NaN.
        if np.isnan(score):
            score = 0

        return -score

    checkpoint_file = os.path.join(output_dir, "skopt.state")
    try:
        state = skopt.load(checkpoint_file)
    except FileNotFoundError:
        x0 = None
        y0 = None
        objective_sequence = count()
    else:
        x0 = state.x_iters
        y0 = state.func_vals
        objective_sequence = count(len(x0))
        n_calls -= len(x0)
        n_random_starts -= len(x0)
        if n_random_starts < 0:
            n_random_starts = 0

    gp_minimize(
        objective,
        dimensions,
        x0=x0,
        y0=y0,
        n_initial_points=n_random_starts,
        n_calls=n_calls,
        n_jobs=n_jobs,
        callback=[
            partial(progress, output_dir),
            DeltaYStopper(1e-6),
            CheckpointSaver(checkpoint_file, store_objective=False),
        ],
    )
    print("Done!")
