from pathlib import Path

from ludwig.api import LudwigModel


def run(config_path: str | Path, dataset_path: str | Path):
    model = LudwigModel(str(config_path))
    model.train(dataset=str(dataset_path))
