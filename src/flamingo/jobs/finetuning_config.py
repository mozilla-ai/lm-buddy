from peft import LoraConfig
from pydantic import Field, root_validator, validator

from flamingo.integrations.huggingface import (
    AutoModelConfig,
    AutoTokenizerConfig,
    DatasetConfig,
    QuantizationConfig,
    TrainerConfig,
)
from flamingo.integrations.wandb import WandbRunConfig
from flamingo.types import BaseFlamingoConfig


class FinetuningRayConfig(BaseFlamingoConfig):
    """Misc settings passed to Ray train for finetuning.

    Includes information for scaling, checkpointing, and runtime storage.
    """

    use_gpu: bool = True
    num_workers: int | None = None
    storage_path: str | None = None  # TODO: This should be set globally somehow


class FinetuningJobConfig(BaseFlamingoConfig):
    """Configuration to submit an LLM finetuning job."""

    model: AutoModelConfig
    dataset: DatasetConfig
    tokenizer: AutoTokenizerConfig
    quantization: QuantizationConfig | None = None
    adapter: LoraConfig | None = None  # TODO: Create own dataclass here
    tracking: WandbRunConfig | None = None
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    ray: FinetuningRayConfig = Field(default_factory=FinetuningRayConfig)

    @root_validator(pre=True)
    def ensure_tokenizer_config(cls, values):
        """Set the tokenizer to the model path when not explicitly provided."""
        if values.get("tokenizer", None) is None:
            match values["model"]:
                case str() as model_path:
                    values["tokenizer"] = model_path
                case dict() as model_data:
                    values["tokenizer"] = model_data["path"]
                case AutoModelConfig() as model_config:
                    values["tokenizer"] = model_config.path
                # No fallback necessary, downstream validation will flag invalid model types
        return values

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
