import json
import random
from collections import defaultdict

import click

from hector_ml.click_helpers import smart_open


@click.command()
@click.option("--negative", type=click.FloatRange(0.0, 1.0), default="1")
@click.option("--positive", type=click.FloatRange(0.0, 1.0), default="1")
@click.argument("input_graphs", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_graphs", type=click.Path(dir_okay=False, writable=True))
def main(negative: float, positive: float, input_graphs: str, output_graphs: str):
    """Downsample manifestation points."""
    rates = {False: negative, True: positive}
    counts = defaultdict(int)
    with smart_open(input_graphs, "r") as input_file, smart_open(
        output_graphs, "wt"
    ) as output_file:
        for line in input_file:
            nldata = json.loads(line)
            mp_labels = dict.fromkeys(nldata["graph"]["manifestation_nodes"])
            for node in nldata["nodes"]:
                node_id = node["id"]
                if node_id in mp_labels:
                    mp_labels[node_id] = bool(node.get("label"))
            mp_by_label = defaultdict(list)
            for mp, label in mp_labels.items():
                mp_by_label[label].append(mp)
            final = []
            for k, v in mp_by_label.items():
                selection = random.sample(v, int(len(v) * rates[k]))
                counts[k] += len(selection)
                final.extend(selection)
            final.sort()
            if final:
                nldata["graph"]["manifestation_nodes"] = final
                print(json.dumps(nldata), file=output_file)
    print(dict(counts))
