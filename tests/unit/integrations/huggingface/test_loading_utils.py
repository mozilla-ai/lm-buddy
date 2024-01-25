from unittest import mock

from datasets import Dataset, DatasetDict

from flamingo.integrations.huggingface import (
    DatasetConfig,
    load_and_split_dataset,
    load_dataset_from_config,
)
from flamingo.integrations.wandb import WandbArtifactConfig


def test_dataset_loading(resources_dir):
    xyz_dataset_path = resources_dir / "datasets" / "xyz.hf"

    # The mock function is imported inside the `loading_utils` module
    # so we mock that path rather than where it is defined
    with mock.patch(
        "flamingo.integrations.huggingface.loading_utils.get_artifact_filesystem_path",
        return_value=xyz_dataset_path,
    ):
        artifact = WandbArtifactConfig(name="xyz-dataset")
        dataset_config = DatasetConfig(load_from=artifact, test_size=0.2, seed=0)

        dataset = load_dataset_from_config(dataset_config)
        assert type(dataset) is Dataset

        datasets = load_and_split_dataset(dataset_config)
        assert type(datasets) is DatasetDict
        assert "train" in datasets and "test" in datasets
