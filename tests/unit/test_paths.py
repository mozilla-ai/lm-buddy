import pytest
from pydantic import TypeAdapter, ValidationError

from lm_buddy.paths import AssetPath, AssetSource


def test_asset_path_validation():
    # Imbues the LoadableAssetPath type with Pydantic validation methods
    adapter_cls = TypeAdapter(AssetPath)

    repo_path = adapter_cls.validate_python("hf://repo_id")
    assert repo_path.scheme == AssetSource.HUGGINGFACE

    file_path = adapter_cls.validate_python("file:///absolute/path")
    assert file_path.scheme == AssetSource.FILE

    wandb_path = adapter_cls.validate_python("wandb://entity/project/name:version")
    assert wandb_path.scheme == AssetSource.WANDB

    with pytest.raises(ValidationError):
        adapter_cls.validate_python("hf://bad...repo_id")
    with pytest.raises(ValidationError):
        adapter_cls.validate_python("file://not/absolute")
