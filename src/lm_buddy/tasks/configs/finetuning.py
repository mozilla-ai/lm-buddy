from pydantic import Field

from lm_buddy.integrations.huggingface import (
    AdapterConfig,
    AutoModelConfig,
    AutoTokenizerConfig,
    QuantizationConfig,
    TextDatasetConfig,
    TrainerConfig,
)
from lm_buddy.integrations.wandb import WandbRunConfig
from lm_buddy.tasks.configs import LMBuddyTaskConfig
from lm_buddy.types import BaseLMBuddyConfig


class FinetuningRayConfig(BaseLMBuddyConfig):
    """Misc settings passed to Ray train for finetuning.

    Includes information for scaling, checkpointing, and runtime storage.
    """

    use_gpu: bool = True
    num_workers: int | None = None
    storage_path: str | None = None  # TODO: This should be set globally somehow


class FinetuningTaskConfig(LMBuddyTaskConfig):
    """Configuration for an LLM finetuning task."""

    model: AutoModelConfig
    dataset: TextDatasetConfig
    tokenizer: AutoTokenizerConfig
    quantization: QuantizationConfig | None = None
    adapter: AdapterConfig | None = None
    tracking: WandbRunConfig | None = None
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    ray: FinetuningRayConfig = Field(default_factory=FinetuningRayConfig)
