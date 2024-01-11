from flamingo.types import BaseFlamingoConfig, SerializableTorchDtype


class TrainerConfig(BaseFlamingoConfig):
    """Configuration for a HuggingFace trainer/training arguments."""

    max_seq_length: int | None = None
    num_train_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    torch_dtype: SerializableTorchDtype = None
    evaluation_strategy: str = "epoch"
    eval_steps: float | None = None
    logging_strategy: str = "steps"
    logging_steps: float = 100
    save_strategy: str = "steps"
    save_steps: int = 500
