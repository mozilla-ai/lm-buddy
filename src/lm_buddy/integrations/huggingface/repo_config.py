from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import field_validator

from lm_buddy.types import BaseLMBuddyConfig


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


class HuggingFaceRepoConfig(BaseLMBuddyConfig):
    """Configuration for a HuggingFace Hub repository."""

    __match_args__ = ("repo_id", "revision")

    repo_id: str
    revision: str | None = None

    @field_validator("repo_id", mode="after")
    def validate_repo_id(cls, x: str):
        if not is_valid_huggingface_repo_id(x):
            raise ValueError(f"{x} is not a valid HuggingFace repo ID.")
        return x
