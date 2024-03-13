from pydantic import model_validator

from lm_buddy.paths import HuggingFaceRepoID, LoadableAssetPath
from lm_buddy.types import BaseLMBuddyConfig

DEFAULT_TEXT_FIELD: str = "text"


class DatasetConfig(BaseLMBuddyConfig):
    """Base configuration to load a HuggingFace dataset."""

    path: LoadableAssetPath
    split: str | None = None
    test_size: float | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def validate_split_if_huggingface_repo(cls, config: "DatasetConfig"):
        """
        Ensure a  `split` is provided when loading a HuggingFace dataset directly from HF Hub.
        This makes it such that the `load_dataset` function returns the type `Dataset`
        instead of `DatasetDict`, which makes some of the downstream logic easier.
        """
        path = config.path
        split = config.split
        if split is None and isinstance(path, HuggingFaceRepoID):
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
