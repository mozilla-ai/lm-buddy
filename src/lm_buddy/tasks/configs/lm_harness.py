from typing import Literal

from pydantic import conlist, model_validator

from lm_buddy.integrations.huggingface import (
    AutoModelConfig,
    QuantizationConfig,
)
from lm_buddy.integrations.vllm import InferenceServerConfig
from lm_buddy.integrations.wandb import WandbRunConfig
from lm_buddy.types import BaseLMBuddyConfig


class LocalChatCompletionsConfig(BaseLMBuddyConfig):
    """Configuration for a "local-chat-completions" model in lm-harness.

    The "local-chat-completions" model is powered by a self-hosted inference server,
    specified as an `InferenceServerConfig`. Additional arguments are also provided
    to control the tokenizer type and generation parameters.
    """

    inference: InferenceServerConfig
    truncate: bool = False
    max_tokens: int = 256
    tokenizer_backend: Literal["huggingface", "tiktoken"] = "huggingface"

    @model_validator(mode="after")
    def validate_inference_engine(cls, config: "LocalChatCompletionsConfig"):
        if config.inference.engine is None:
            raise ValueError(
                "Inference config `engine` must be provided for use in "
                "lm-harness 'local-chat-completions' model."
            )
        return config


class LMHarnessEvaluationConfig(BaseLMBuddyConfig):
    """Misc settings provided to an lm-harness evaluation job."""

    tasks: conlist(str, min_length=1)
    batch_size: int | None = None
    num_fewshot: int | None = None
    limit: int | float | None = None


class LMHarnessTaskConfig(BaseLMBuddyConfig):
    """Configuration to run an lm-evaluation-harness evaluation task."""

    model: AutoModelConfig | LocalChatCompletionsConfig
    evaluation: LMHarnessEvaluationConfig
    quantization: QuantizationConfig | None = None
    tracking: WandbRunConfig | None = None
