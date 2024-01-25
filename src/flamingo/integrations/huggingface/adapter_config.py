from peft import LoraConfig, TaskType
from pydantic import Field

from flamingo.types import BaseFlamingoConfig

# TODO(RD2024-68): Support other PEFT adapters


class LoraAdapterConfig(BaseFlamingoConfig):
    """Configuration for a LORA adapter for LLM finetuning.

    Settings derived from the HuggingFace implementation class `LoraConfig`
    (https://huggingface.co/docs/peft/conceptual_guides/lora).
    """

    task_type: TaskType = Field(
        default=TaskType.CAUSAL_LM,
        description=(
            "Training task the adapter is being used for. "
            "Must always be specified (default: TaskType.CAUSAL_LM)",
        ),
    )
    r: int | None = None
    target_modules: str | list[str] | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    fan_in_fan_out: bool | None = None
    bias: str | None = None
    modules_to_save: list[str] | None = None
    init_lora_weights: bool | str | None = None
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    rank_pattern: dict | None = None
    alpha_pattern: dict | None = None
    loftq_config: dict | None = None

    def as_huggingface(self) -> LoraConfig:
        """Return the HuggingFace implementation of `LoraConfig`."""
        # Filter None values to use HF defaults when not specified
        args = {k: v for k, v in self.dict().items() if v is not None}
        return LoraConfig(**args)
