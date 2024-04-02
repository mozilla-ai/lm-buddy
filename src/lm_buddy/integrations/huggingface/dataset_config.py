from pydantic import model_validator

from lm_buddy.paths import AssetPath, PathPrefix
from lm_buddy.types import BaseLMBuddyConfig

DEFAULT_TEXT_FIELD: str = "text"


class DatasetConfig(BaseLMBuddyConfig):
    """Base configuration to load a HuggingFace dataset."""

    path: AssetPath
    split: str | None = None
    test_size: float | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def validate_split_if_huggingface_path(cls, config: "DatasetConfig"):
        """
        Ensure a  `split` is provided when loading a HuggingFace dataset directly from HF Hub.
        This makes it such that the `load_dataset` function returns the type `Dataset`
        instead of `DatasetDict`, which makes some of the downstream logic easier.
        """
        if config.split is None and config.path.startswith(PathPrefix.HUGGINGFACE):
            raise ValueError(
                "A `split` must be specified when loading a dataset directly from HuggingFace."
            )
        return config


class TextDatasetConfig(DatasetConfig):
    """Settings passed to load a HuggingFace text dataset.

    Inherits fields from the the base `DatasetConfig`.
    The dataset should contain a single text column named by the `text_field` parameter.

    A `prompt_template` can be provided to format columns of the dataset.
    The formatted prompt is added to the dataset as the `text_field`.
    """

    text_field: str = DEFAULT_TEXT_FIELD
    prompt_template: str | None = None
