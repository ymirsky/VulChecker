from __future__ import annotations

import fileinput
import json
import math
from contextlib import ExitStack

import click

from hector_ml.click_helpers import smart_open


def check_dict_finite(data, context):
    all_okay = True
    for k, v in data.items():
        if isinstance(v, float) and not math.isfinite(v):
            print(f"{context}:{k} has non-finite value {v}.")
            all_okay = False
    return all_okay


def check_nldata(nldata: dict, check_labels: bool, context: str):
    all_okay = True
    labels = dict.fromkeys(nldata["graph"]["manifestation_nodes"])
    if not labels:
        print(f"{context} has no manifestation nodes.")
        all_okay = False
    for node in nldata["nodes"]:
        if node["id"] in labels:
            labels[node["id"]] = node.get("label")
        if not check_dict_finite(node, f"{context}:{node['id']}"):
            all_okay = False
    for link in nldata["links"]:
        if not check_dict_finite(node, f"{context}:{link['source']}->{link['target']}"):
            all_okay = False
    if check_labels and labels and True not in labels.values():
        print(f"{context} has no positive labels.")
        all_okay = False
    return all_okay


@click.command()
@click.option("--check-labels", is_flag=True, help="Check that graphs have labels.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True),
    help="File where good data will be written.",
)
@click.argument("datasets", nargs=-1, type=click.Path(dir_okay=False, exists=True))
@click.pass_context
def main(
    ctx: click.Context,
    check_labels: bool,
    output: str | None,
    datasets: tuple[str, ...],
):
    all_okay = True
    with ExitStack() as stack:
        lines = stack.enter_context(
            fileinput.input(datasets, mode="rb", openhook=fileinput.hook_compressed)
        )
        if output is not None:
            output = stack.enter_context(smart_open(output, "wb"))
        for line in lines:
            nldata = json.loads(line)
            if check_nldata(
                nldata, check_labels, f"{lines.filename()}:{lines.filelineno()}"
            ):
                if output is not None:
                    output.write(line)
            else:
                all_okay = False

    ctx.exit(0 if all_okay else 1)
