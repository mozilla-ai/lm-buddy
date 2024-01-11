from pathlib import Path

from pydantic.dataclasses import dataclass

from flamingo.integrations.huggingface.utils import is_valid_huggingface_model_name


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
    checkpoint: str | None = None

    def __post_init__(self):
        if isinstance(self.name, Path):
            self.name = str(self.name)

        if Path(self.name).is_absolute():
            self.checkpoint = self.name
        else:
            self.checkpoint = None

        if self.checkpoint is None and not is_valid_huggingface_model_name(self.name):
            raise ValueError(f"{self.name} is not a valid checkpoint path or HF model name.")
