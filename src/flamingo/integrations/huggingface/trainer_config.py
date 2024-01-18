from typing import Any

from flamingo.types import BaseFlamingoConfig


class TrainerConfig(BaseFlamingoConfig):
    """Configuration for a HuggingFace trainer/training arguments.

    This mainly encompasses arguments passed to the HuggingFace `TrainingArguments` class,
    but also contains some additional parameters for the `Trainer` or `SFTTrainer` classes.
    """

    max_seq_length: int | None = None
    num_train_epochs: float | None = None
    per_device_train_batch_size: int | None = None
    per_device_eval_batch_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    gradient_accumulation_steps: int | None = None
    gradient_checkpointing: bool | None = None
    evaluation_strategy: str | None = None
    eval_steps: float | None = None
    logging_strategy: str | None = None
    logging_steps: float | None = None
    save_strategy: str | None = None
    save_steps: int | None = None

    def training_args(self) -> dict[str, Any]:
        """Return the arguments to the HuggingFace `TrainingArguments` class."""
        excluded_keys = ["max_seq_length"]
        return self.dict(exclude=excluded_keys)
