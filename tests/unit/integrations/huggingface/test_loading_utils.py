import pytest
import torch
from datasets import Dataset, DatasetDict

from flamingo.integrations.huggingface import AutoModelConfig, DatasetConfig, HuggingFaceAssetLoader
from flamingo.integrations.wandb import ArtifactType, WandbArtifactConfig, build_directory_artifact
from tests.test_utils import FakeWandbArtifactLoader


@pytest.fixture
def xyz_dataset_artifact(resources_dir):
    dataset_path = resources_dir / "datasets" / "xyz"
    return build_directory_artifact(
        artifact_name="xyz-dataset",
        artifact_type=ArtifactType.DATASET,
        dir_path=dataset_path,
        reference=True,
    )


@pytest.fixture
def bert_model_artifact(resources_dir):
    model_path = resources_dir / "models" / "bert_tiniest"
    return build_directory_artifact(
        artifact_name="bert-tiniest-model",
        artifact_type=ArtifactType.MODEL,
        dir_path=model_path,
        reference=True,
    )


def test_dataset_loading(xyz_dataset_artifact):
    # Preload fake artifact for testing
    artifact_loader = FakeWandbArtifactLoader()
    artifact_loader.log_artifact(xyz_dataset_artifact)
    hf_loader = HuggingFaceAssetLoader(artifact_loader)

    artifact_config = WandbArtifactConfig(name=xyz_dataset_artifact.name, project="project")
    dataset_config = DatasetConfig(load_from=artifact_config, test_size=0.2, seed=0)

    dataset = hf_loader.load_dataset(dataset_config)
    assert type(dataset) is Dataset

    datasets = hf_loader.load_and_split_dataset(dataset_config)
    assert type(datasets) is DatasetDict
    assert "train" in datasets and "test" in datasets


def test_model_loading(bert_model_artifact):
    # Preload fake artifact for testing
    artifact_loader = FakeWandbArtifactLoader()
    artifact_loader.log_artifact(bert_model_artifact)
    hf_loader = HuggingFaceAssetLoader(artifact_loader)

    artifact_config = WandbArtifactConfig(name=bert_model_artifact.name, project="project")
    model_config = AutoModelConfig(load_from=artifact_config, torch_dtype=torch.bfloat16)

    bert_config = hf_loader.load_pretrained_config(model_config)
    bert_model = hf_loader.load_pretrained_model(model_config)
    assert bert_config._name_or_path == bert_model.name_or_path
    assert bert_model.dtype == torch.bfloat16
