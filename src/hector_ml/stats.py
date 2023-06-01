import csv
import os.path
import pickle
from contextlib import ExitStack
from functools import partial
from itertools import islice

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from more_itertools import ilen
from sklearn import metrics
from structure2vec.graph_collate import graph_collate

from hector_ml.click_helpers import smart_open
from hector_ml.graphs import GraphDataset, JSONGraphs, sink_graph_infos
from hector_ml.model import Predictor


def get_source_line(root, filename, line):
    if root is None or filename is None or line is None or line < 1:
        return ""
    try:
        with open(os.path.join(root, filename), "rt") as f:
            return next(islice(f, line - 1, line)).strip()
    except (FileNotFoundError, StopIteration):
        return ""


@click.command()
@click.option(
    "--device", type=torch.device, default="cuda", help="Device on which to run."
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
    "--predictions-csv",
    type=click.Path(dir_okay=False, writable=True),
    help="File where CSV prediction information will be written.",
)
@click.option(
    "--dump",
    type=click.Path(dir_okay=False, writable=True),
    help="File where outputs will be written.",
)
@click.option(
    "--source-dir",
    type=click.Path(file_okay=False, exists=True),
    help="Directory containing original source files.",
)
@click.option(
    "--roc-file",
    type=click.Path(dir_okay=False, writable=True),
    help="File where ROC plot will be saved.",
)
@click.option(
    "--exec-only", is_flag=True,
    help="For making predicitons on data with no labels.",
)
@click.argument("output_dir", type=click.Path(file_okay=False, exists=True))
@click.argument("testing_graphs", type=click.Path(dir_okay=False, exists=True))
def main(
    device,
    batch_size,
    predictions_csv,
    dump,
    source_dir,
    roc_file,
    exec_only,
    output_dir,
    testing_graphs,
):
    with ExitStack() as stack:
        if device.type == "cuda":
            stack.enter_context(torch.cuda.device(device))

        predictor = Predictor.from_checkpoint_dir(output_dir, map_location=device)

        testing_f = stack.enter_context(smart_open(testing_graphs, "rt"))
        test_dataset = GraphDataset(
            JSONGraphs(testing_f),
            predictor.node_features,
            predictor.edge_features,
            predictor.depth_limit,
        )

        # load from cache if exists to save time
        #test_cache_path = str(testing_graphs) + ".cache.pkl"
        #if os.path.isfile(test_cache_path):
        #    print("Loading test from cache")
        #    with open(test_cache_path, "rb") as handle:
        #        test_dataset = pickle.load(handle)
        #else:
        print("Loading test")
        test_dataset = list(test_dataset)
        #    print("Saving test as cache")
        #    with open(test_cache_path, "wb") as handle:
        #        pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        n_observations = ilen(test_dataset)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=partial(graph_collate, mode=predictor.reduction_mode),
        )

        final_embeddings = np.zeros(
            (n_observations, predictor.model_params["embedding_dimensions"]),
            dtype=np.float32,
        )
        log_probs = np.zeros(
            (n_observations, predictor.model_params["n_classes"]), dtype=np.float64
        )
        final_labels = np.zeros(n_observations, dtype=np.uint8)
        final_index = 0
        with torch.no_grad():
            for data, labels in test_loader:
                this_batch_size = len(labels)
                data.features[0] -= predictor.model.mean
                data.features[0] /= predictor.model.std
                embeddings = predictor.model.embedding(data)
                final_embeddings[
                    final_index : final_index + this_batch_size
                ] = embeddings
                log_probs[
                    final_index : final_index + this_batch_size
                ] = torch.nn.functional.log_softmax(
                    predictor.model.logits(
                        predictor.model.classifier(embeddings).to(torch.float64)
                    ),
                    dim=-1,
                )
                final_labels[final_index : final_index + this_batch_size] = labels
                final_index += this_batch_size

        final_scores = log_probs[:, 1]

        if exec_only:
            final_predictions = final_scores
        else:
            fpr, tpr, thresh = metrics.roc_curve(final_labels, final_scores)
            decision_index = np.argmax(tpr - fpr)
            final_predictions = (final_scores >= thresh[decision_index]).astype(final_labels.dtype)

            if dump:
                np.savez(
                    dump,
                    cwe=predictor.cwe,
                    labels=final_labels,
                    embeddings=final_embeddings,
                    log_probs=log_probs,
                    scores=final_scores,
                    roc_fpr=fpr,
                    roc_tpr=tpr,
                    roc_thresh=thresh,
                    decision_index=decision_index,
                    predictions=final_predictions,
                )
    
            auc = metrics.auc(fpr, tpr)
            if roc_file:
                plt.plot(fpr, tpr)
                plt.xlim([0.0, 1.05])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False positive rate")
                plt.ylabel("True positive rate")
                plt.title(f"HECTOR - CWE{predictor.cwe} (AUC = {auc:.3})")
                plt.savefig(roc_file)

            print("Decision threshold:", thresh[decision_index])
            print("True positive rate:", tpr[decision_index])
            print("False positive rate:", fpr[decision_index])
            print("Area under ROC:", auc)

        if predictions_csv:
            with smart_open(predictions_csv, "wt", newline="") as csv_f:
                testing_f.seek(0)
                writer = csv.writer(csv_f)
                writer.writerow(
                    ["File", "Line", "Score", "Prediction", "Label", "Source"]
                )
                for graph_info, label, score, prediction in zip(
                    sink_graph_infos(JSONGraphs(testing_f)),
                    final_labels,
                    final_scores,
                    final_predictions,
                ):
                    writer.writerow(
                        [
                            graph_info["filename"],
                            graph_info["line_number"],
                            score,
                            prediction,
                            label,
                            get_source_line(
                                source_dir,
                                graph_info["filename"],
                                graph_info["line_number"],
                            ),
                        ]
                    )
