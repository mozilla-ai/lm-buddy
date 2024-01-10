from typing import Any

from flamingo.integrations.wandb import WandbEnvironment

import wandb
from wandb.apis.public import Run


def get_wandb_summary(env: WandbEnvironment) -> dict[str, Any]:
    """Get the summary dictionary attached to a W&B run."""
    run = _resolve_wandb_run(env)
    return dict(run.summary)


def update_wandb_summary(env: WandbEnvironment, metrics: dict[str, Any]) -> None:
    """Update a run's summary with the provided metrics."""
    run = _resolve_wandb_run(env)
    run.summary.update(metrics)
    run.update()


def _resolve_wandb_run(env: WandbEnvironment) -> Run:
    """Resolve a WandB run object from the provided environment settings.

    An exception is raised if a Run cannot be found,
    or if multiple runs exist in scope with the same name.
    """
    api = wandb.Api()
    base_path = "/".join(x for x in (env.entity, env.project) if x)
    if env.run_id is not None:
        full_path = f"{base_path}/{env.run_id}"
        return api.run(full_path)
    else:
        match [run for run in api.runs(base_path) if run.name == env.name]:
            case []:
                raise RuntimeError(f"No WandB runs found at {base_path}/{env.name}")
            case [Run(), _]:
                raise RuntimeError(f"Multiple WandB runs found at {base_path}/{env.name}")
            case [Run()] as mr:
                # we have a single one, hurray
                return mr[0]
