import contextlib
from typing import Any

import wandb
from wandb.apis.public import Run as ApiRun

from flamingo.integrations.wandb import WandbRunConfig
from flamingo.types import BaseFlamingoConfig


@contextlib.contextmanager
def wandb_init_from_config(
    config: WandbRunConfig,
    *,
    resume: str | None = None,
    job_type: str | None = None,
    parameters: BaseFlamingoConfig | None = None,
):
    """Initialize a W&B run from the internal run configuration.

    This method can be entered as a context manager similar to `wandb.init` as follows:

    ```
    with wandb_init_from_config(run_config, resume="must") as run:
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
        config=parameters.dict() if parameters else None,
        resume=resume,
        job_type=job_type,
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
