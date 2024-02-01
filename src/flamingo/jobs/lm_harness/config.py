import datetime
from typing import Literal

from pydantic import Field, conlist, model_validator

from flamingo.integrations.huggingface import (
    AutoModelConfig,
    QuantizationConfig,
)
from flamingo.integrations.vllm import InferenceServerConfig
from flamingo.integrations.wandb import WandbRunConfig
from flamingo.types import BaseFlamingoConfig


class LocalChatCompletionsConfig(BaseFlamingoConfig):
    """Configuration for a "local-chat-completions" model in lm-harness.

    The "local-chat-completions" model is powered by a self-hosted inference server,
    specified as an `InferenceServerConfig`. Additional arguments are also provided
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


class LMHarnessRayConfig(BaseFlamingoConfig):
    """Misc settings for Ray compute in the LM harness job."""

    num_cpus: int | float = 1
    num_gpus: int | float = 1
    timeout: datetime.timedelta | None = None


class LMHarnessEvaluatorConfig(BaseFlamingoConfig):
    """Misc settings provided to an lm-harness evaluation job."""

    tasks: conlist(str, min_length=1)
    batch_size: int | None = None
    num_fewshot: int | None = None
    limit: int | float | None = None


class LMHarnessJobConfig(BaseFlamingoConfig):
    """Configuration to run an lm-evaluation-harness evaluation job."""

    model: AutoModelConfig | LocalChatCompletionsConfig
    evaluator: LMHarnessEvaluatorConfig
    quantization: QuantizationConfig | None = None
    tracking: WandbRunConfig | None = None
    ray: LMHarnessRayConfig = Field(default_factory=LMHarnessRayConfig)
