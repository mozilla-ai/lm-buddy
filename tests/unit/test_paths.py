import pytest
from pydantic import TypeAdapter, ValidationError

from lm_buddy.integrations.wandb import WandbArtifactConfig
from lm_buddy.paths import FilesystemPath, HuggingFaceRepoID, LoadableAssetPath


def test_asset_path_validation():
    asset_path_adapter = TypeAdapter(LoadableAssetPath)

    repo_id = asset_path_adapter.validate_python("repo_id")
    assert isinstance(repo_id, HuggingFaceRepoID)

    filesystem_path = asset_path_adapter.validate_python("/absolute/path")
    assert isinstance(filesystem_path, FilesystemPath)

    artifact_config = WandbArtifactConfig(name="artifact", project="project")
    artifact_config = asset_path_adapter.validate_python(artifact_config)
    assert isinstance(artifact_config, WandbArtifactConfig)

    with pytest.raises(ValidationError):
        asset_path_adapter.validate_python("bad...repo_id")
    with pytest.raises(ValidationError):
        asset_path_adapter.validate_python(120850120)
