from peft import LoraConfig
from pydantic import validator
from ray.train import ScalingConfig

from flamingo.integrations.huggingface import QuantizationConfig
from flamingo.integrations.huggingface.utils import is_valid_huggingface_model_name
from flamingo.integrations.wandb import WandbEnvironment
from flamingo.jobs.configs import BaseJobConfig
from flamingo.types import SerializableTorchDtype


class FinetuningJobConfig(BaseJobConfig):
    """Configuration to submit an LLM finetuning job."""

    model: str
    dataset: str
    tokenizer: str | None = None
    # Training
    max_seq_length: int | None = None
    num_train_epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    torch_dtype: SerializableTorchDtype = None
    # Logging
    evaluation_strategy: str = "epoch"
    eval_steps: float | None = None
    logging_strategy: str = "steps"
    logging_steps: float = 100
    save_strategy: str = "steps"
    save_steps: int = 500
    wandb_env: WandbEnvironment | None = None
    # Lora/quantization
    lora_config: LoraConfig | None = None  # TODO: Create our own config type
    quantization_config: QuantizationConfig | None = None
    # Cluster
    storage_path: str | None = None
    scaling_config: ScalingConfig | None = None  # TODO: Create our own config type

    @validator("model")
    def _validate_modelname(cls, v):  # noqa: N805
        if is_valid_huggingface_model_name(v):
            return v
        else:
            raise (ValueError(f"`{v}` is not a valid HuggingFace model name."))
