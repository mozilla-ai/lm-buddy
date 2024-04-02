import re
from enum import Enum
from typing import Annotated

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import AfterValidator


class PathPrefix(str, Enum):
    FILE = "file://"
    WANDB = "wandb://"
    HUGGINGFACE = "hf://"


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


def validate_asset_path(path: str) -> str:
    if path.startswith((PathPrefix.FILE, PathPrefix.WANDB)):
        return path
    elif path.startswith(PathPrefix.HUGGINGFACE):
        raw_path = strip_path_prefix(path)
        if not is_valid_huggingface_repo_id(raw_path):
            raise ValueError(f"{path} is not a valid HuggingFace repo ID.")
    else:
        allowed = {x.value for x in PathPrefix}
        raise ValueError(f"{path} does not start with one of the allowed prefixes: {allowed}")


AssetPath = Annotated[str, AfterValidator(lambda x: validate_asset_path(x))]
"""String representing the name/path for loading a HuggingFace asset.

The path begins with one of the allowed `PathPrefix`s that determine how to load the asset.
"""


def strip_path_prefix(path: AssetPath) -> str:
    pattern = "^\w+\:\/\/"  # Matches '{prefix}://' at start of string
    return re.sub(pattern, "", path)
