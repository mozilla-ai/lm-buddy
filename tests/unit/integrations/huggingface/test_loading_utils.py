import pytest
from datasets import Dataset, DatasetDict

from flamingo.integrations.huggingface import (
    DatasetConfig,
    load_and_split_dataset,
    load_dataset_from_config,
)
from flamingo.integrations.wandb import ArtifactType, WandbArtifactConfig, build_directory_artifact
from tests.test_utils import FakeWandbArtifactLoader


@pytest.fixture
def xyz_dataset_artifact(resources_dir):
    xyz_dataset_path = resources_dir / "datasets" / "xyz"
    return build_directory_artifact(
        artifact_name="xyz-dataset",
        artifact_type=ArtifactType.DATASET,
        dir_path=xyz_dataset_path,
        reference=True,
    )


def test_dataset_loading(xyz_dataset_artifact):
    # Log fake artifacr for test
    artifact_loader = FakeWandbArtifactLoader()
    artifact_loader.log_artifact(xyz_dataset_artifact)

    artifact_config = WandbArtifactConfig(name=xyz_dataset_artifact.name, project="project")
    dataset_config = DatasetConfig(load_from=artifact_config, test_size=0.2, seed=0)

    dataset = load_dataset_from_config(dataset_config, artifact_loader)
    assert type(dataset) is Dataset

    datasets = load_and_split_dataset(dataset_config, artifact_loader)
    assert type(datasets) is DatasetDict
    assert "train" in datasets and "test" in datasets
