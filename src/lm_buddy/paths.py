import re
from enum import Enum
from pathlib import Path
from typing import Annotated

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import AfterValidator


class PathPrefix(str, Enum):
    FILE = "file://"
    HUGGINGFACE = "hf://"
    WANDB = "wandb://"


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


def strip_path_prefix(path: str) -> str:
    """Strip 'prefix://' from the start of a string."""
    pattern = "^\w+\:\/\/"
    return re.sub(pattern, "", path)


def validate_path_prefix(path: str, prefix: PathPrefix) -> None:
    if not path.startswith(prefix):
        raise ValueError(f"{path} does not start with the expected prefix {prefix}.")
    return path


def validate_file_path(path: str) -> str:
    path = validate_path_prefix(path, PathPrefix.FILE)
    raw_path = strip_path_prefix(path)
    if not Path(raw_path).is_absolute():
        raise ValueError(f"{raw_path} is not an absolute file path.")
    return path


def validate_hf_path(path: str) -> str:
    path = validate_path_prefix(path, PathPrefix.HUGGINGFACE)
    raw_path = strip_path_prefix(path)
    if not is_valid_huggingface_repo_id(raw_path):
        raise ValueError(f"{raw_path} is not a valid HuggingFace repo ID.")
    return path


def validate_wandb_path(path: str) -> str:
    path = validate_path_prefix(path, PathPrefix.WANDB)
    return path


FilePath = Annotated[str, AfterValidator(lambda x: validate_file_path(x))]
"""Path string of the form 'file:///absolute/path/to/asset'."""

HuggingFacePath = Annotated[str, AfterValidator(lambda x: validate_hf_path(x))]
"""Path string of the form 'hf://repo-name'."""

WandbArtifactPath = Annotated[str, AfterValidator(lambda x: validate_wandb_path(x))]
"""Path string of the form 'wandb://entity/project/name:version'."""

AssetPath = FilePath | HuggingFacePath | WandbArtifactPath
"""String representing the name/path for loading a HuggingFace asset.

The path begins with one of the allowed `PathPrefix`s that determine how to load the asset.
"""


def format_file_path(path: str | Path) -> FilePath:
    return f"{PathPrefix.FILE}{path}"


def format_huggingface_path(repo: str) -> HuggingFacePath:
    return f"{PathPrefix.HUGGINGFACE}{repo}"


def format_wandb_path(
    name: str,
    project: str,
    entity: str | None = None,
    version: str = "latest",
) -> WandbArtifactPath:
    base_path = "/".join(x for x in [entity, project, name] if x is not None)
    return f"{PathPrefix.WANDB}{base_path}:{version}"
