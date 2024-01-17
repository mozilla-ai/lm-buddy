import datetime

from pydantic import Field

from flamingo.integrations.huggingface import AutoModelConfig, QuantizationConfig
from flamingo.integrations.wandb import WandbRunConfig
from flamingo.types import BaseFlamingoConfig


class RayComputeSettings(BaseFlamingoConfig):
    """Misc settings for Ray compute in the LM harness job."""

    use_gpu: bool = True
    num_workers: int = 1
    timeout: datetime.timedelta | None = None


class LMHarnessEvaluatorSettings(BaseFlamingoConfig):
    """Misc settings provided to an lm-harness evaluation job."""

    tasks: list[str]
    batch_size: int | None = None
    num_fewshot: int | None = None
    limit: int | float | None = None


class LMHarnessJobConfig(BaseFlamingoConfig):
    """Configuration to run an lm-evaluation-harness evaluation job."""

    model: AutoModelConfig
    evaluator: LMHarnessEvaluatorSettings
    quantization: QuantizationConfig | None = None
    tracking: WandbRunConfig | None = None
    ray: RayComputeSettings = Field(default_factory=RayComputeSettings)
