"""
This file is used to provide fixtures for the test session accessible to all Flamingo submodules.
"""
from pathlib import Path

import pytest

from flamingo.integrations.wandb import ArtifactType, build_directory_artifact


@pytest.fixture(scope="session")
def examples_dir():
    return Path(__file__).parents[1] / "examples"


@pytest.fixture(scope="session")
def resources_dir():
    return Path(__file__).parent / "resources"


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
def text_dataset_artifact(resources_dir):
    dataset_path = resources_dir / "datasets" / "tiny_shakespeare"
    return build_directory_artifact(
        artifact_name="tiny-shakespeare-dataset",
        artifact_type=ArtifactType.DATASET,
        dir_path=dataset_path,
        reference=True,
    )


@pytest.fixture
def llm_model_artifact(resources_dir):
    model_path = resources_dir / "models" / "fake_gpt2"
    return build_directory_artifact(
        artifact_name="fake-gpt2-model",
        artifact_type=ArtifactType.MODEL,
        dir_path=model_path,
        reference=True,
    )
