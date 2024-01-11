from dataclasses import InitVar
from pathlib import Path

from huggingface_hub.utils import HFValidationError, validate_repo_id
from pydantic.dataclasses import dataclass


def is_valid_huggingface_model_name(s: str):
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


@dataclass
class ModelNameOrCheckpointPath:
    """
    This class is explicitly used to validate if a string is
    a valid HuggingFace model or can be used as a checkpoint.

    Checkpoint will be automatically assigned if it's a valid checkpoint;
    it will be None if it's not valid.
    """

    # explictly needed for matching
    __match_args__ = ("name", "checkpoint")

    name: str
    checkpoint: InitVar[str | None] = None

    def __post_init__(self, checkpoint):
        if isinstance(self.name, Path):
            self.name = str(self.name)

        if Path(self.name).is_absolute():
            self.checkpoint = self.name
        else:
            self.checkpoint = None

        if self.checkpoint is None and not is_valid_huggingface_model_name(self.name):
            raise (ValueError(f"{self.name} is not a valid checkpoint path or HF model name"))
