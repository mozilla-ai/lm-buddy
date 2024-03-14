from pathlib import Path
from typing import Annotated, Any

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import BeforeValidator

from lm_buddy.integrations.wandb import WandbArtifactConfig
from lm_buddy.types import BaseLMBuddyConfig


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
        case str() if Path(x).is_absolute():
            return Path(x)
        case str() if is_valid_huggingface_repo_id(x):
            return HuggingFaceRepoID(repo_id=x)
        case str():
            raise ValueError(f"{x} is neither a valid HuggingFace repo ID or an absolute path.")
        case _:
            # Handled by downstream "after" validators
            return x


LoadableAssetPath = Annotated[
    Path | HuggingFaceRepoID | WandbArtifactConfig,
    BeforeValidator(lambda x: validate_asset_path(x)),
]
"""A value that can be resolved to a name/path for loading a HuggingFace asset.

During validation, the following conversions occur:
    - Strings representing an absolute path (beginning with a '/') are converted to `Path`s
    - Other strings are converted to `HuggingFaceRepoID`s
    - Other values are passed to the default Pydantic validators
"""
