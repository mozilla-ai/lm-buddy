from pydantic import field_validator, model_validator

from lm_buddy.integrations.huggingface import HuggingFaceRepoConfig, convert_string_to_repo_config
from lm_buddy.integrations.wandb import WandbArtifactConfig
from lm_buddy.types import BaseLMBuddyConfig

DEFAULT_TEXT_FIELD: str = "text"


class DatasetConfig(BaseLMBuddyConfig):
    """Base configuration to load a HuggingFace dataset."""

    load_from: HuggingFaceRepoConfig | WandbArtifactConfig
    split: str | None = None
    test_size: float | None = None
    seed: int | None = None

    _validate_load_from_string = field_validator("load_from", mode="before")(
        convert_string_to_repo_config
    )

    @model_validator(mode="after")
    def validate_split_if_huggingface_repo(cls, config: "DatasetConfig"):
        """
        Ensure a  `split` is provided when loading a HuggingFace dataset directly from HF Hub.
        This makes it such that the `load_dataset` function returns the type `Dataset`
        instead of `DatasetDict`, which makes some of the downstream logic easier.
        """
        load_from = config.load_from
        split = config.split
        if split is None and isinstance(load_from, HuggingFaceRepoConfig):
            raise ValueError(
                "A `split` must be specified when loading a dataset directly from HuggingFace."
            )
        return config


class TextDatasetConfig(DatasetConfig):
    """Settings passed to load a HuggingFace text dataset.

    Inherits fields from the the base `DatasetConfig`.
    The dataset should contain a single text column named by the `text_field` parameter.
    """

    text_field: str = DEFAULT_TEXT_FIELD
