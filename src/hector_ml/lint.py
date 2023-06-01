import collections
import csv
import heapq
import json
import operator
from pathlib import Path
from typing import Optional, TextIO, Tuple, Union

import click
import networkx as nx
import torch

from hector_ml.configure import HectorConfig, detect_build_system
from hector_ml.model import Predictor
from hector_ml.predict import Prediction, find_vulns
from hector_ml.preprocess import (
    CALL_OPERATION,
    MANIFESTATION_POINT_TAG,
    ROOT_CAUSE_TAG,
    add_invariant_graph_features,
    merge_edges,
    relativize_file_names,
    remove_llvm_internal_functions,
    translate_categorical_features,
)

CWE_PIPELINE_ALIASES = {191: 190, 122: 121}


class LintOutput:
    def __init__(self, fh):
        self.fh = fh

    def __call__(self, predictor: Predictor, prediction: Prediction):
        mp = prediction.manifestation_point
        print(
            f"{mp.file_name}:{mp.line_number}:W:CWE{predictor.cwe} manifests here "
            f"with estimated probability {prediction.probability:.3}",
            file=self.fh,
        )
        if prediction.root_cause:
            rc = prediction.root_cause
            print(
                f"{rc.file_name}:{rc.line_number}:I:CWE{predictor.cwe} "
                "root cause here",
                file=self.fh,
            )


class CsvOutput:
    def __init__(self, fh):
        self.writer = csv.writer(fh)
        self.writer.writerow(
            ["CWE", "manif_file", "manif_line", "rc_file", "rc_line", "probability"]
        )

    def __call__(self, predictor: Predictor, prediction: Prediction):
        self.writer.writerow(
            [
                predictor.cwe,
                prediction.manifestation_point.file_name,
                prediction.manifestation_point.line_number,
                prediction.root_cause.file_name if prediction.root_cause else "",
                prediction.root_cause.line_number if prediction.root_cause else "",
                prediction.probability,
            ]
        )


OUTPUT_HANDLERS = {
    "lint": LintOutput,
    "csv": CsvOutput,
}


@click.command()
@click.option(
    "--device", type=torch.device, default="cuda", help="Device on which to run."
)
@click.option(
    "--llap-lib-dir",
    type=click.Path(file_okay=False, exists=True),
    default="/usr/local/lib",
    show_default=True,
    help="Directory containing HECTOR opt passes.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(0, 1),
    default=0.5,
    show_default=True,
    help="Decision threshold probability.",
)
@click.option(
    "--top",
    "top_k",
    metavar="K",
    type=click.IntRange(1, None),
    help="Show only K most-likely vulnerabilities (per CWE).",
)
@click.option(
    "--output",
    type=click.File("wt"),
    default="-",
    show_default=True,
    help="File where output will be written.",
)
@click.option(
    "--output-format",
    type=click.Choice(list(OUTPUT_HANDLERS)),
    default="lint",
    show_default=True,
    help="Output style",
)
@click.argument(
    "source_dir",
    type=click.Path(file_okay=False, exists=True, writable=True),
    default=".",
)
@click.argument("target")
@click.argument(
    "model_dirs",
    metavar="MODEL_DIR",
    nargs=-1,
    type=click.Path(file_okay=False, exists=True),
)
@click.pass_context
def main(
    ctx: click.Context,
    device: torch.device,
    llap_lib_dir: Union[bytes, str],
    threshold: float,
    top_k: Optional[int],
    output: TextIO,
    output_format: str,
    source_dir: Union[bytes, str],
    target: str,
    model_dirs: Tuple[Union[bytes, str], ...],
):
    """Lint-check a codebase using HECTOR."""
    llap_lib_dir = Path(llap_lib_dir).resolve()
    source_dir = Path(source_dir).resolve()

    predictors = [
        Predictor.from_checkpoint_dir(model_dir, map_location=device)
        for model_dir in model_dirs
    ]
    pipelines = collections.defaultdict(list)
    for predictor in predictors:
        pipelines[CWE_PIPELINE_ALIASES.get(predictor.cwe, predictor.cwe)].append(
            predictor
        )

    build_system = detect_build_system(source_dir)
    if build_system is None:
        ctx.fail("No build system detected.")
    hector_dir = source_dir / "hector_build"
    hector_dir.mkdir(exist_ok=True)
    hector_config = HectorConfig(
        source_dir,
        source_dir,
        hector_dir,
        llap_lib_dir,
        build_system,
        target,
        pipelines,
    )
    hector_config.make()

    output_handler = OUTPUT_HANDLERS[output_format](output)
    for pipeline, predictors in pipelines.items():
        with open(hector_config.hector_dir / f"hector-{pipeline}.json", "r") as f:
            raw_pdg = nx.node_link_graph(json.load(f), directed=True)

        # N.B. Using components of preprocess_graph here to avoid redundant work.
        remove_llvm_internal_functions(raw_pdg)
        relativize_file_names(raw_pdg, source_dir)
        raw_pdg = merge_edges(raw_pdg)
        for predictor in predictors:
            cwe_pdg = raw_pdg.copy()
            translate_categorical_features(cwe_pdg, predictor.indexes)
            add_invariant_graph_features(
                cwe_pdg,
                manifestation_index=predictor.indexes["tag"][MANIFESTATION_POINT_TAG],
                call_index=predictor.indexes["operation"][CALL_OPERATION],
                root_cause_index=predictor.indexes["tag"][ROOT_CAUSE_TAG],
                edge_type_indexes=predictor.indexes["type"],
            )

            vulns = find_vulns(cwe_pdg, predictor, threshold)
            if top_k is not None:
                vulns = heapq.nlargest(
                    top_k, vulns, key=operator.attrgetter("probability")
                )
            for prediction in vulns:
                output_handler(predictor, prediction)
