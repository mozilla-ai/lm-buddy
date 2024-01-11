import datetime
from pathlib import Path
from typing import Any

from pydantic import root_validator, validator

from flamingo.integrations.huggingface import ModelNameOrCheckpointPath, QuantizationConfig
from flamingo.integrations.wandb import WandbEnvironment
from flamingo.jobs.configs import BaseJobConfig
from flamingo.types import SerializableTorchDtype


class LMHarnessJobConfig(BaseJobConfig):
    """Configuration to run an lm-evaluation-harness evaluation job.

    This job loads an existing checkpoint path
    from Ray storage to run evaluation against, OR a huggingface Model.
    and logs the evaluation results to W&B.
    When a W&B config is specified, the job will attempt to resolve a checkpoint path
    associated with that run.

    This can be manually overwritten by specifying the `model_name_or_path` variable
    which will take prescedence over the W&B checkpoint path.
    """

    class Config:
        validate_assignment = True

    tasks: list[str]
    model_name_or_path: str | Path | ModelNameOrCheckpointPath | None = None
    batch_size: int | None = None
    num_fewshot: int | None = None
    limit: int | float | None = None
    trust_remote_code: bool = False
    torch_dtype: SerializableTorchDtype = None
    quantization_config: QuantizationConfig | None = None
    wandb_env: WandbEnvironment | None = None
    num_cpus: int = 1
    num_gpus: int = 1
    timeout: datetime.timedelta | None = None

    @validator("model_name_or_path", pre=True, always=True)
    def _validate_modelname(cls, v):  # noqa: N805
        """
        happens pre-validation and makes sure we correctly set a typed var for matching.
        """
        if isinstance(v, dict):
            return ModelNameOrCheckpointPath(**v)
        elif v is None:
            return None
        else:
            return ModelNameOrCheckpointPath(name=v)

    @root_validator(pre=False)
    @classmethod
    def _validate_modelname_or_checkpoint(cls, values) -> Any:
        """
        Primarily logic to infer if a passed value is a HuggingFace model,
        checkpoint for resuming, or not.
        """
        mnp = values.get("model_name_or_path")
        wandb_env = values.get("wandb_env")

        # fairly complex logic here:
        # we're matching on the structure of the passed args

        match (mnp, wandb_env):
            case (None, None):
                raise (ValueError("Either `model_name_or_path` or `wandb_env` must be provided."))

            case (None, WandbEnvironment()):
                print(
                    "no model name or checkpoint passed; will attempt to run from passed wandb run."
                )
            case (ModelNameOrCheckpointPath() as x, _):
                print(
                    "will ignore passed information from a wandb run, "
                    f"if present, and prefer loading from: {x.name}"
                )
            case _:
                raise (ValueError(f"{mnp} is not a valid HuggingFaceModel or checkpoint path."))

        return values
