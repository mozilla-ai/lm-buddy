import datetime
from pathlib import Path

from pydantic import validator

from flamingo.integrations.huggingface import ModelNameOrCheckpointPath, QuantizationConfig
from flamingo.jobs.configs import BaseJobConfig
from flamingo.types import SerializableTorchDtype


class LMHarnessJobConfig(BaseJobConfig):
    """Configuration to run an lm-evaluation-harness evaluation job.

    This job loads an existing checkpoint path from Ray storage to run evaluation against,
    OR a huggingface Model and logs the evaluation results to W&B.

    This can be manually overwritten by specifying the `model_name_or_path` variable
    which will take prescedence over the W&B checkpoint path.
    """

    class Config:
        validate_assignment = True

    tasks: list[str]
    batch_size: int | None = None
    num_fewshot: int | None = None
    limit: int | float | None = None
    trust_remote_code: bool = False
    torch_dtype: SerializableTorchDtype = None
    model_name_or_path: str | Path | ModelNameOrCheckpointPath | None = None
    quantization: QuantizationConfig | None = None
    num_cpus: int = 1
    num_gpus: int = 1
    timeout: datetime.timedelta | None = None

    @validator("model_name_or_path", pre=True, always=True)
    def _validate_model_name_or_path(cls, v):
        if isinstance(v, dict):
            return ModelNameOrCheckpointPath(**v)
        elif v is None:
            return None
        else:
            return ModelNameOrCheckpointPath(name=v)
