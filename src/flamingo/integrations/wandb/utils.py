from typing import Any

import wandb
from wandb.apis.public import Run as ApiRun

from flamingo.integrations.wandb import WandbRunConfig


def get_wandb_api_run(run_config: WandbRunConfig) -> ApiRun:
    """Retrieve a run from the W&B API."""
    api = wandb.Api()
    return api.run(run_config.get_wandb_path())


def get_wandb_summary(run_config: WandbRunConfig) -> dict[str, Any]:
    """Get the summary dictionary attached to a W&B run."""
    run = get_wandb_api_run(run_config)
    return dict(run.summary)


def update_wandb_summary(run_config: WandbRunConfig, metrics: dict[str, Any]) -> None:
    """Update a run's summary with the provided metrics."""
    run = get_wandb_api_run(run_config)
    run.summary.update(metrics)
    run.update()
