from pydantic import ValidationError, root_validator, validator

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

    @root_validator()
    def validate_split_if_huggingface_repo(cls, values):
        """
        Ensure a  `split` is provided when loading a HuggingFace dataset directly from HF Hub.
        This makes it such that the `load_dataset` function returns the type `Dataset`
        instead of `DatasetDict`, which makes some of the downstream logic easier.
        """
        load_from = values["load_from"]
        split = values.get("split")
        if split is None and isinstance(load_from, HuggingFaceRepoConfig):
            raise ValidationError(
                "A `split` must be specified when loading a dataset directly from HuggingFace."
            )
        return values


class TextDatasetConfig(DatasetConfig):
    """Settings passed to load a HuggingFace text dataset.

    Inherits fields from the the base `DatasetConfig`.
    The dataset should contain a single text column named by the `text_field` parameter.
    """

    text_field: str = DEFAULT_TEXT_FIELD
