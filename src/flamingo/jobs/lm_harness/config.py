import datetime
from typing import Literal

from pydantic import Field, conlist

from flamingo.integrations.huggingface import (
    AutoModelConfig,
    HuggingFacePathReference,
    QuantizationConfig,
)
from flamingo.integrations.wandb import WandbRunConfig
from flamingo.types import BaseFlamingoConfig


class LocalChatCompletionsConfig(BaseFlamingoConfig):
    """Configuration for a "local-chat-completions" model in lm-harness.

    The "local-chat-completions" model connects to a locally hosted inference server
    provided by the `base_url`. It is also necessary to provide the `engine` of
    the locally hosted inference server, which can be a raw string, HF repo config,
    or a W&B artifact config.
    """

    __match_args__ = ("base_url", "engine", "tokenizer_backend")

    base_url: str
    engine: str | HuggingFacePathReference
    tokenizer_backend: Literal["huggingface", "tiktoken"] = "huggingface"


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
