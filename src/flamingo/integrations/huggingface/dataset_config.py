from pydantic import validator

from flamingo.integrations.huggingface.utils import repo_id_validator
from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig


class DatasetConfig(BaseFlamingoConfig):
    """Settings passed to load a HuggingFace dataset."""

    path: str | WandbArtifactConfig
    split: str | None = None
    text_field: str = "text"
    test_size: float | None = None
    seed: int | None = None

    _path_validator = validator("path", allow_reuse=True, pre=True)(repo_id_validator)
