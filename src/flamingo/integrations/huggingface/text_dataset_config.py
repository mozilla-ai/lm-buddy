from pydantic import validator

from flamingo.integrations.huggingface.utils import repo_id_validator
from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig

DEFAULT_TEXT_FIELD: str = "text"


class TextDatasetConfig(BaseFlamingoConfig):
    """Settings passed to load a HuggingFace text dataset.

    The dataset should contain a single text column named by the `text_field` parameter.
    """

    path: str | WandbArtifactConfig
    split: str | None = None
    text_field: str = DEFAULT_TEXT_FIELD
    test_size: float | None = None
    seed: int | None = None

    _path_validator = validator("path", allow_reuse=True, pre=True)(repo_id_validator)
