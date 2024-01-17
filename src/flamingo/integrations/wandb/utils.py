from typing import Any

import wandb
from wandb.apis.public import Run

from flamingo.integrations.wandb import WandbRunLink


def get_wandb_run(env: WandbRunLink) -> Run:
    """Retrieve a run from the W&B API."""
    api = wandb.Api()
    return api.run(env.wandb_path)


def get_wandb_summary(env: WandbRunLink) -> dict[str, Any]:
    """Get the summary dictionary attached to a W&B run."""
    run = get_wandb_run(env)
    return dict(run.summary)


def update_wandb_summary(env: WandbRunLink, metrics: dict[str, Any]) -> None:
    """Update a run's summary with the provided metrics."""
    run = get_wandb_run(env)
    run.summary.update(metrics)
    run.update()
