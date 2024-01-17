from typing import Any

from peft import LoraConfig
from pydantic import Field, validator

from flamingo.integrations.huggingface import (
    AutoModelConfig,
    AutoTokenizerConfig,
    DatasetConfig,
    QuantizationConfig,
    TrainerConfig,
)
from flamingo.integrations.wandb import WandbRunConfig
from flamingo.types import BaseFlamingoConfig


class RayTrainConfig(BaseFlamingoConfig):
    """Misc settings passed to Ray train.

    Includes information for scaling, checkpointing, and runtime storage.
    """

    use_gpu: bool = True
    num_workers: int | None = None
    storage_path: str | None = None

    def get_scaling_args(self) -> dict[str, Any]:
        args = dict(use_gpu=self.use_gpu, num_workers=self.num_workers)
        return {k: v for k, v in args.items() if v is not None}


class FinetuningJobConfig(BaseFlamingoConfig):
    """Configuration to submit an LLM finetuning job."""

    model: AutoModelConfig
    dataset: DatasetConfig
    tokenizer: AutoTokenizerConfig | None = None
    quantization: QuantizationConfig | None = None
    adapter: LoraConfig | None = None  # TODO: Create own dataclass here
    tracking: WandbRunConfig | None = None
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    ray: RayTrainConfig = Field(default_factory=RayTrainConfig)

    @validator("model", pre=True, always=True)
    def validate_model_arg(cls, x):
        """Allow for passing just a path string as the model argument."""
        if isinstance(x, str):
            return AutoModelConfig(path=x)
        return x

    @validator("dataset", pre=True, always=True)
    def validate_dataset_arg(cls, x):
        """Allow for passing just a path string as the dataset argument."""
        if isinstance(x, str):
            return DatasetConfig(path=x)
        return x

    @validator("tokenizer", pre=True, always=True)
    def validate_tokenizer_arg(cls, x):
        """Allow for passing just a path string as the tokenizer argument."""
        if isinstance(x, str):
            return AutoTokenizerConfig(path=x)
        return x
