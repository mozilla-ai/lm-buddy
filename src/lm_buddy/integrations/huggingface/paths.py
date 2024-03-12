from pathlib import Path

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import field_validator

from lm_buddy.integrations.wandb import WandbArtifactConfig
from lm_buddy.types import BaseLMBuddyConfig


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


class HuggingFaceRepoID(BaseLMBuddyConfig):
    """Repository name on the HuggingFace Hub."""

    __match_args__ = ("repo_id",)

    repo_id: str

    @field_validator("repo_id", mode="after")
    def validate_repo_id(cls, x: str):
        if not is_valid_huggingface_repo_id(x):
            raise ValueError(f"{x} is not a valid HuggingFace repo ID.")
        return x


HuggingFaceAssetPath = Path | HuggingFaceRepoID | WandbArtifactConfig
"""Type that can be resolved to a path for loading a HuggingFace asset."""
