import pytest
from pydantic import TypeAdapter, ValidationError

from lm_buddy.paths import AssetPath, PathScheme


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


def test_scheme_identifcation():
    file_path = AssetPath.from_file_path("/path/to/file")
    assert file_path.scheme == PathScheme.FILE

    hf_path = AssetPath.from_huggingface_repo("distilgpt2")
    assert hf_path.scheme == PathScheme.HUGGINGFACE

    wandb_path = AssetPath.from_wandb("artifact", "project")
    assert wandb_path.scheme == PathScheme.WANDB


def test_strip_prefix():
    file_path = AssetPath.from_file_path("/path/to/file")
    assert file_path.strip_prefix() == "/path/to/file"

    hf_path = AssetPath.from_huggingface_repo("distilgpt2")
    assert hf_path.strip_prefix() == "distilgpt2"

    wandb_path = AssetPath.from_wandb("name", "project", version="v0")
    assert wandb_path.strip_prefix() == "project/name:v0"
