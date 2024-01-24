from pydantic import validator

from flamingo.integrations.huggingface import HuggingFaceRepoConfig, convert_to_repo_config
from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig

DEFAULT_TEXT_FIELD: str = "text"


class TextDatasetConfig(BaseFlamingoConfig):
    """Settings passed to load a HuggingFace text dataset.

    The dataset should contain a single text column named by the `text_field` parameter.
    """

    load_from: HuggingFaceRepoConfig | WandbArtifactConfig
    split: str | None = None
    text_field: str = DEFAULT_TEXT_FIELD
    test_size: float | None = None
    seed: int | None = None

    _validate_load_from = validator("load_from", pre=True, allow_reuse=True)(convert_to_repo_config)
