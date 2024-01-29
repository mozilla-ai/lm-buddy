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


def test_from_wandb_path():
    valid_path_with_entity = "entity/project/name:latest"
    config_with_entity = WandbArtifactConfig.from_wandb_path(valid_path_with_entity)
    assert config_with_entity.name == "name"
    assert config_with_entity.project == "project"
    assert config_with_entity.version == "latest"
    assert config_with_entity.entity == "entity"

    valid_path_without_entity = "project/name:latest"
    config_without_entity = WandbArtifactConfig.from_wandb_path(valid_path_without_entity)
    assert config_without_entity.name == "name"
    assert config_without_entity.project == "project"
    assert config_without_entity.version == "latest"
    assert config_without_entity.entity is None

    with pytest.raises(ValueError):
        WandbArtifactConfig.from_wandb_path("entity/project/name")  # No version
    with pytest.raises(ValueError):
        WandbArtifactConfig.from_wandb_path("entity/project/name/version")  # Bad delimiter
