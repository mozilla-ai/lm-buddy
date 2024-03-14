from pathlib import Path
from typing import Annotated, Any

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import BeforeValidator

from lm_buddy.integrations.wandb import WandbArtifactConfig


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
            return x
        case str():
            raise ValueError(f"{x} is neither a valid HuggingFace repo ID or an absolute path.")
        case _:
            # Handled by downstream "after" validators
            return x


LoadableAssetPath = Annotated[
    str | Path | WandbArtifactConfig,
    BeforeValidator(lambda x: validate_asset_path(x)),
]
"""A value that can be resolved to a path for loading an asset from disk.

During validation, the following conversions occur:
- Strings representing an absolute path (beginning with a '/') are converted to `Path` instances
- Other strings are validated as HuggingFace repo IDs
- Other objects are validated as `WandbArtifactConfig` instances
"""
