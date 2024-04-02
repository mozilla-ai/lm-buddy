from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urlparse

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class PathScheme(str, Enum):
    FILE = "file"
    HUGGINGFACE = "hf"
    WANDB = "wandb"


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


def validate_asset_path(path: str) -> "AssetPath":
    match urlparse(path):
        case ParseResult(scheme=PathScheme.FILE, path=file_path):
            if not Path(file_path).is_absolute():
                raise ValueError(f"{file_path} is not an absolute file path.")
        case ParseResult(scheme=PathScheme.HUGGINGFACE, netloc=repo_id):
            if not is_valid_huggingface_repo_id(repo_id):
                raise ValueError(f"{repo_id} is not a valid HuggingFace repo ID.")
        case ParseResult(scheme=PathScheme.WANDB):
            # TODO: Validate the W&B path structure?
            pass
        case _:
            allowed = {x.value for x in PathScheme}
            raise ValueError(f"{path} does not begin with an allowed prefix: {allowed}")
    return AssetPath(path)


class AssetPath(str):
    """String representing the name/path for loading an asset.

    The path begins with one of the allowed `PathScheme`s that determine how to load the asset.
    """

    @classmethod
    def from_file(cls, path: str | Path) -> "AssetPath":
        path = Path(path).absolute()
        return cls(f"{PathScheme.FILE}://{path}")

    @classmethod
    def from_huggingface_repo(cls, repo_id: str) -> "AssetPath":
        return cls(f"{PathScheme.HUGGINGFACE}://{repo_id}")

    @classmethod
    def from_wandb(
        cls,
        name: str,
        project: str,
        entity: str | None = None,
        version: str = "latest",
    ) -> "AssetPath":
        base_path = "/".join(x for x in [entity, project, name] if x is not None)
        return cls(f"{PathScheme.WANDB}://{base_path}:{version}")

    @property
    def scheme(self) -> PathScheme:
        scheme = urlparse(self).scheme
        return PathScheme(scheme)

    def strip_prefix(self) -> str:
        """Strip 'scheme://' from the start of a path string."""
        scheme = self.scheme
        return self.replace(f"{scheme}://", "")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Defining Pydantic validation logic on a custom class.

        Reference: https://docs.pydantic.dev/latest/concepts/types/
        """
        return core_schema.no_info_after_validator_function(validate_asset_path, handler(str))
