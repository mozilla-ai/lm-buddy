from typing import Any

from datasets import DatasetDict, load_dataset
from huggingface_hub.utils import HFValidationError, validate_repo_id


def repo_id_validator(x: Any):
    if isinstance(x, str) and not is_valid_huggingface_repo_id(x):
        raise ValueError(f"{x} is not a valid HuggingFace repo ID.")
    return x


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


def load_and_split_dataset(
    path: str,
    *,
    split: str | None = None,
    test_size: float | None,
    seed: int | None = None,
) -> DatasetDict:
    dataset = load_dataset(path, split=split)
    if test_size is not None:
        datasets = dataset.train_test_split(test_size=test_size, seed=seed)
    else:
        datasets = DatasetDict({"train": dataset})
    return datasets
