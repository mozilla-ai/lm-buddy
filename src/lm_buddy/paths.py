from enum import Enum
from pathlib import Path
from typing import Annotated

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import AfterValidator, UrlConstraints
from pydantic_core import Url


def strip_url_scheme(url: Url) -> str:
    """Strip the 'scheme://' portion from a `Url`.

    For example, 'file:///path/to/file' will return '/path/to/file'.
    """
    return str(url).replace(f"{url.scheme}://", "")


def is_valid_huggingface_repo_id(s: str):
    """
    Simple test to check if an HF model is valid using HuggingFace's tools.
    Sadly, theirs throws an exception and has no return.

    Args:
        s: string to test.
    """
    try:
        validate_repo_id(s)
        return True
    except HFValidationError:
        return False


def validate_asset_path(path: "AssetPath") -> "AssetPath":
    if path.scheme == AssetSource.HUGGINGFACE:
        repo_id = strip_url_scheme(path)
        if not is_valid_huggingface_repo_id(repo_id):
            raise ValueError(f"{repo_id} is not a valid HuggingFace repo ID.")
    elif path.scheme == AssetSource.FILE:
        file_path = strip_url_scheme(path)
        if not Path(file_path).is_absolute():
            raise ValueError(f"{file_path} is not an absolute file path.")
    return path


class AssetSource(str, Enum):
    FILE = "file"
    WANDB = "wandb"
    HUGGINGFACE = "hf"


AssetPath = Annotated[
    Url,
    UrlConstraints(allowed_schemes=[x.value for x in AssetSource]),
    AfterValidator(validate_asset_path),
]
"""Type representing the path for loading HuggingFace asset.

The path is represented by a `Url` with scheme specified by the type of `AssetSource`.

The `AssetSource` controls the interpretation of the path:
- `hf://` indicates the path is a HuggingFace Hub repository name
- `file://` indicates a path on the local filesystem (must be absolute with a leading '/')
- `wandb://` indicates the path is the `entity/project/name:version` of a W&B artifact
"""


def build_huggingface_asset_path(repo_id: str) -> AssetPath:
    return AssetPath.build(scheme=AssetSource.HUGGINGFACE, host=repo_id)


def build_file_asset_path(file: str | Path) -> AssetPath:
    return AssetPath.build(scheme=AssetSource.FILE, host="", path=str(file))


def build_wandb_asset_path(
    name: str,
    project: str,
    entity: str | None = None,
    version: str = "latest",
) -> AssetPath:
    path = f"{project}/{name}:{version}"
    if entity is not None:
        path = f"{entity}/{path}"
    return AssetPath.build(scheme=AssetSource.WANDB, host=path)
