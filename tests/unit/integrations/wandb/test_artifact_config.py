import pytest

from flamingo.integrations.wandb import WandbArtifactConfig


@pytest.fixture
def wandb_artifact_config():
    return WandbArtifactConfig(
        name="artifact-name",
        version="latest",
        project="research",
        entity="team",
    )


def test_serde_round_trip(wandb_artifact_config):
    assert WandbArtifactConfig.parse_raw(wandb_artifact_config.json()) == wandb_artifact_config


def test_wandb_path(wandb_artifact_config):
    assert wandb_artifact_config.wandb_path() == "team/research/artifact-name:latest"
