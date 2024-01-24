from unittest import mock

from datasets import Dataset, DatasetDict

from flamingo.integrations.huggingface import (
    DatasetConfig,
    load_and_split_dataset,
    load_dataset_from_config,
)
from flamingo.integrations.wandb import WandbArtifactConfig


def test_load_dataset_from_config(resources_folder):
    xyz_dataset_path = resources_folder / "datasets" / "xyz.hf"

    def fake_resolve_path(*args, **kwargs):
        return str(xyz_dataset_path), None

    # Intentionally mocking `resolve_loadable_path` because mocking had trouble finding
    # `get_artifact_filesystem_path`, likely due to import ordering in the package
    with mock.patch(
        "flamingo.integrations.huggingface.loading_utils.resolve_loadable_path",
        fake_resolve_path,
    ):
        artifact = WandbArtifactConfig(name="xyz-dataset")
        config = DatasetConfig(load_from=artifact)
        dataset = load_dataset_from_config(config)
        assert type(dataset) is Dataset


def test_load_and_split_dataset(resources_folder):
    xyz_dataset_path = resources_folder / "datasets" / "xyz.hf"

    def fake_resolve_path(*args, **kwargs):
        return str(xyz_dataset_path), None

    # Intentionally mocking `resolve_loadable_path` because mocking had trouble finding
    # `get_artifact_filesystem_path`, likely due to import ordering in the package
    with mock.patch(
        "flamingo.integrations.huggingface.loading_utils.resolve_loadable_path",
        fake_resolve_path,
    ):
        artifact = WandbArtifactConfig(name="xyz-dataset")
        config = DatasetConfig(load_from=artifact, test_size=0.2, seed=0)
        datasets = load_and_split_dataset(config)
        assert type(datasets) is DatasetDict
        assert "train" in datasets and "test" in datasets
