from typing import Any

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic import validator

from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig


class LocalServerConfig(BaseFlamingoConfig):
    """Configuration for a HuggingFace Hub repository."""

    __match_args__ = ("path")

    path: str

    @validator("path", pre=True)
    def validate_repo_id(cls, x):
        if isinstance(x, str) and "v1/completions" not in x:
            raise ValueError(f"{x} is not a valid vLLM OpenAI style inference server.")
        return x


LoadFromLocalConfig = LocalServerConfig | WandbArtifactConfig
"""Config that can be resolved to a local server path or a weightsandbiases path."""
