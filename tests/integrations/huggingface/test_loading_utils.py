from unittest import mock

from datasets import DatasetDict

from flamingo.integrations.huggingface import DatasetConfig, load_dataset_from_config
from flamingo.integrations.wandb import WandbArtifactConfig


def test_load_dataset_from_config(resources_folder):
    xyz_dataset_path = resources_folder / "datasets" / "xyz.hf"

    with mock.patch(
        "flamingo.integrations.wandb.artifact_utils.get_artifact_filesystem_path",
        return_value=xyz_dataset_path,
    ):
        artifact_config = WandbArtifactConfig(name="xyz-dataset")
        dataset_config = DatasetConfig(load_from=artifact_config, split="train", test_size=0.5)
        dataset = load_dataset_from_config(dataset_config)
        assert type(dataset) is DatasetDict
        assert "test" in dataset and "train" in dataset
