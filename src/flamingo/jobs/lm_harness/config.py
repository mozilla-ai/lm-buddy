import datetime

from pydantic import Field, conlist

from flamingo.integrations.huggingface import AutoModelConfig, QuantizationConfig
from flamingo.integrations.vllm import InferenceServerConfig
from flamingo.integrations.wandb import WandbRunConfig
from flamingo.types import BaseFlamingoConfig


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

    model: AutoModelConfig | InferenceServerConfig
    evaluator: LMHarnessEvaluatorConfig
    quantization: QuantizationConfig | None = None
    tracking: WandbRunConfig | None = None
    ray: LMHarnessRayConfig = Field(default_factory=LMHarnessRayConfig)
