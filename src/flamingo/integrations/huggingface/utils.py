from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id


def repo_id_validator(x: Any):
    if isinstance(x, str) and not is_valid_huggingface_repo_id(x):
        raise ValueError(f"{x} is not a valid HuggingFace repo ID.")
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
