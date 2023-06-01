import importlib
import logging

import click


class CLI(click.MultiCommand):
    COMMANDS = [
        "augmentation",
        "compile_for_train",
        "configure",
        "cross_validation",
        "feature_stats",
        "hyperopt",
        "lint",
        "predict",
        "preprocess",
        "sample_data",
        "stats",
        "train",
        "train_test_split",
        "validate_data",
        "visualize",
    ]

    def list_commands(self, ctx):
        return self.COMMANDS

    def get_command(self, ctx, name):
        if name not in self.COMMANDS:
            raise KeyError(name)
        return importlib.import_module(f"hector_ml.{name}").main


@click.command(cls=CLI)
def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s",
    )
