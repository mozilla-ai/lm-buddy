from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import validator

from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig


def convert_string_to_repo_config(x: Any):
    if isinstance(x, str):
        return HuggingFaceRepoConfig(repo_id=x)
    return x


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


class HuggingFaceRepoConfig(BaseFlamingoConfig):
    """Configuration for a HuggingFace Hub repository."""

    __match_args__ = ("repo_id", "revision")

    repo_id: str
    revision: str | None = None

    @validator("repo_id", pre=True)
    def validate_repo_id(cls, x):
        if isinstance(x, str) and not is_valid_huggingface_repo_id(x):
            raise ValueError(f"{x} is not a valid HuggingFace repo ID.")
        return x


LoadFromConfig = HuggingFaceRepoConfig | WandbArtifactConfig
"""Config that can be resolved to a HuggingFace name/path."""
