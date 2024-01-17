from typing import Any

from flamingo.types import BaseFlamingoConfig


class TrainerConfig(BaseFlamingoConfig):
    """Configuration for a HuggingFace trainer/training arguments.

    This mainly encompasses arguments passed to the HuggingFace `TrainingArguments` class,
    but also contains some additional parameters for the `Trainer` or `SFTTrainer` classes.
    """

    max_seq_length: int | None = None
    num_train_epochs: int | None = None
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

    def get_training_args(self) -> dict[str, Any]:
        args = dict(
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=self.gradient_checkpointing,
            weight_decay=self.weight_decay,
            evaluation_strategy=self.evaluation_strategy,
            eval_steps=self.eval_steps,
            logging_strategy=self.logging_strategy,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
        )
        # Only return non-None values so we use the HuggingFace defaults when not specified
        return {k: v for k, v in args.items() if v is not None}
