from peft import LoraConfig
from pydantic import validator
from ray.train import ScalingConfig

from flamingo.integrations.huggingface import QuantizationConfig
from flamingo.integrations.huggingface.trainer_config import TrainerConfig
from flamingo.integrations.huggingface.utils import is_valid_huggingface_model_name
from flamingo.jobs import BaseJobConfig


class FinetuningJobConfig(BaseJobConfig):
    """Configuration to submit an LLM finetuning job."""

    model: str
    dataset: str
    tokenizer: str | None = None
    trainer: TrainerConfig | None = None
    lora: LoraConfig | None = None  # TODO: Create our own config type
    quantization: QuantizationConfig | None = None
    scaling: ScalingConfig | None = None  # TODO: Create our own config type
    storage_path: str | None = None

    @validator("model")
    def _validate_model_name(cls, v):
        if is_valid_huggingface_model_name(v):
            return v
        else:
            raise ValueError(f"`{v}` is not a valid HuggingFace model name.")
