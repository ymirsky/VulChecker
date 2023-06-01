import click
import numpy as np

from hector_ml.click_helpers import smart_open


@click.command()
@click.option(
    "--test-fraction",
    metavar="FLOAT",
    type=click.FloatRange(0.0, 1.0),
    default=0.1,
    show_default=True,
)
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("train_path", type=click.Path(dir_okay=False, writable=True))
@click.argument("test_path", type=click.Path(dir_okay=False, writable=True))
def main(test_fraction, input_path, train_path, test_path):
    with smart_open(input_path, "rt") as f:
        input_dataset = list(f)

    train_samples = round(len(input_dataset) * (1.0 - test_fraction))

    indexes = np.random.choice(len(input_dataset), len(input_dataset), replace=False)

    with smart_open(train_path, "wt") as f:
        for idx in indexes[:train_samples]:
            f.write(input_dataset[idx])

    with smart_open(test_path, "wt") as f:
        for idx in indexes[train_samples:]:
            f.write(input_dataset[idx])
