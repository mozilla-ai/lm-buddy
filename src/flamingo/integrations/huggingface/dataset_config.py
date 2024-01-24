from pydantic import validator

from flamingo.integrations.huggingface import HuggingFaceRepoConfig, convert_string_to_repo_config
from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig

DEFAULT_TEXT_FIELD: str = "text"


class DatasetConfig(BaseFlamingoConfig):
    """Base configuration to load a HuggingFace dataset."""

    load_from: HuggingFaceRepoConfig | WandbArtifactConfig
    split: str | None = None
    test_size: float | None = None
    seed: int | None = None

    _validate_load_from_string = validator("load_from", pre=True, allow_reuse=True)(
        convert_string_to_repo_config
    )


class TextDatasetConfig(DatasetConfig):
    """Settings passed to load a HuggingFace text dataset.

    Inherits fields from the the base `DatasetConfig`.
    The dataset should contain a single text column named by the `text_field` parameter.
    """

    text_field: str = DEFAULT_TEXT_FIELD
