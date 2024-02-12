import contextlib
from enum import Enum
from typing import Any

import wandb
from wandb.apis.public import Run as ApiRun

from lm_buddy.integrations.wandb import WandbRunConfig
from lm_buddy.types import BaseLMBuddyConfig


class WandbResumeMode(str, Enum):
    """Enumeration of modes for resuming a W&B run.

    This is not an exahustive list of the values that can be passed to the W&B SDK
    (Docs: https://docs.wandb.ai/ref/python/init), but just those commonly used within the package.
    """

    ALLOW = "allow"
    MUST = "must"
    NEVER = "never"


@contextlib.contextmanager
def wandb_init_from_config(
    config: WandbRunConfig,
    *,
    parameters: BaseLMBuddyConfig | None = None,
    resume: WandbResumeMode | None = None,
    job_type: str | None = None,
):
    """Initialize a W&B run from the internal run configuration.

    This method can be entered as a context manager similar to `wandb.init` as follows:

    ```
    with wandb_init_from_config(run_config, resume=WandbResumeMode.MUST) as run:
        # Use the initialized run here
        ...
    ```
    """
    init_kwargs = dict(
        id=config.run_id,
        name=config.name,
        project=config.project,
        entity=config.entity,
        group=config.run_group,
        config=parameters.model_dump(mode="json") if parameters else None,
        job_type=job_type,
        resume=resume,
    )
    with wandb.init(**init_kwargs) as run:
        yield run


def get_wandb_api_run(config: WandbRunConfig) -> ApiRun:
    """Retrieve a run from the W&B API."""
    api = wandb.Api()
    return api.run(config.wandb_path())


def get_wandb_summary(config: WandbRunConfig) -> dict[str, Any]:
    """Get the summary dictionary attached to a W&B run."""
    run = get_wandb_api_run(config)
    return dict(run.summary)


def update_wandb_summary(config: WandbRunConfig, metrics: dict[str, Any]) -> None:
    """Update a run's summary with the provided metrics."""
    run = get_wandb_api_run(config)
    run.summary.update(metrics)
    run.update()
