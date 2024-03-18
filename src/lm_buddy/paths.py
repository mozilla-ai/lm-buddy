from pathlib import Path
from typing import Annotated, Any

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import BeforeValidator

from lm_buddy.integrations.wandb import WandbArtifactConfig
from lm_buddy.types import BaseLMBuddyConfig


class FilePath(BaseLMBuddyConfig):
    """Absolute path to an object on the local filesystem."""

    __match_args__ = ("path",)

    path: Path


class HuggingFaceRepoID(BaseLMBuddyConfig):
    """Repository ID on the HuggingFace Hub."""

    __match_args__ = ("repo_id",)

    repo_id: str


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


def validate_asset_path(x: Any) -> Any:
    match x:
        case Path() as p:
            return FilePath(path=p)
        case str() as s if Path(s).is_absolute():
            return FilePath(path=s)
        case str() as s if is_valid_huggingface_repo_id(s):
            return HuggingFaceRepoID(repo_id=s)
        case str():
            raise ValueError(f"{x} is neither a valid HuggingFace repo ID or an absolute path.")
        case _:
            # Handled by downstream "after" validators
            return x


AssetPath = Annotated[
    FilePath | HuggingFaceRepoID | WandbArtifactConfig,
    BeforeValidator(lambda x: validate_asset_path(x)),
]
"""Union type representing the name/path for loading HuggingFace asset.

The path is represented by either a `FileSystemPath`, a `HuggingFaceRepoID`
or a `WandbArtifactConfig` that can be resolved to a path via the artifact's manifest.

This type is annotated with Pydantic validation logic to convert absolute path strings
to `FilesystemPath`s and other strings to `HuggingFaceRepoID`s.
"""
