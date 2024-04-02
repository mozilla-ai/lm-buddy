import torch
from datasets import Dataset, DatasetDict

from lm_buddy.integrations.huggingface import AutoModelConfig, DatasetConfig, HuggingFaceAssetLoader
from lm_buddy.paths import AssetPath
from tests.utils import FakeArtifactLoader


def test_dataset_loading(xyz_dataset_artifact):
    # Preload fake artifact for testing
    artifact_loader = FakeArtifactLoader()
    artifact_loader.log_artifact(xyz_dataset_artifact)
    hf_loader = HuggingFaceAssetLoader(artifact_loader)

    artifact_path = AssetPath.from_wandb(name=xyz_dataset_artifact.name, project="project")
    dataset_config = DatasetConfig(path=artifact_path, test_size=0.2, seed=0)

    dataset = hf_loader.load_dataset(dataset_config)
    assert type(dataset) is Dataset

    datasets = hf_loader.load_and_split_dataset(dataset_config)
    assert type(datasets) is DatasetDict
    assert "train" in datasets and "test" in datasets


def test_model_loading(llm_model_artifact):
    # Preload fake artifact for testing
    artifact_loader = FakeArtifactLoader()
    artifact_loader.log_artifact(llm_model_artifact)
    hf_loader = HuggingFaceAssetLoader(artifact_loader)

    artifact_path = AssetPath.from_wandb(name=llm_model_artifact.name, project="project")
    model_config = AutoModelConfig(path=artifact_path, torch_dtype=torch.bfloat16)

    hf_config = hf_loader.load_pretrained_config(model_config)
    hf_model = hf_loader.load_pretrained_model(model_config)
    assert hf_config._name_or_path == hf_model.name_or_path
    assert hf_model.dtype == torch.bfloat16
