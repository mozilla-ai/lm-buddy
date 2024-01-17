from pathlib import Path
from typing import Any

import wandb
from wandb.apis.public import Run as ApiRun

from flamingo.integrations.wandb import WandbRunConfig


def get_wandb_api_run(config: WandbRunConfig) -> ApiRun:
    """Retrieve a run from the W&B API."""
    api = wandb.Api()
    return api.run(config.get_wandb_path())


def get_wandb_summary(config: WandbRunConfig) -> dict[str, Any]:
    """Get the summary dictionary attached to a W&B run."""
    run = get_wandb_api_run(config)
    return dict(run.summary)


def update_wandb_summary(config: WandbRunConfig, metrics: dict[str, Any]) -> None:
    """Update a run's summary with the provided metrics."""
    run = get_wandb_api_run(config)
    run.summary.update(metrics)
    run.update()


def get_artifact_filesystem_path(artifact: wandb.Artifact) -> str:
    for entry in artifact.manifest.entries.values():
        if entry.ref.startswith("file://"):
            # TODO: What if there are entries with different base paths in the artifact manifest?
            entry_path = Path(entry.ref.replace("file://", ""))
            return str(entry_path.parent.absolute())
    raise ValueError("Artifact does not contain a filesystem reference.")
