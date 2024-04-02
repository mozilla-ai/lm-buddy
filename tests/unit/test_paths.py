import pytest
from pydantic import TypeAdapter, ValidationError

from lm_buddy.paths import AssetPath


def test_asset_path_validation():
    # Imbues the LoadableAssetPath type with Pydantic validation methods
    adapter_cls = TypeAdapter(AssetPath)

    valid_paths = [
        "hf://repo-name",
        "file:///path/to/file",
        "wandb://entity/project/name:version",
    ]
    for path in valid_paths:
        adapter_cls.validate_python(path)

    invalid_paths = ["hf://bad..name", "random://scheme", 12345]
    for path in invalid_paths:
        with pytest.raises(ValidationError):
            adapter_cls.validate_python(path)
