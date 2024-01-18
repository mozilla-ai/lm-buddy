from pathlib import Path
from typing import Any

import wandb
from wandb.apis.public import Run as ApiRun

from flamingo.integrations.wandb import WandbArtifactConfig, WandbArtifactLoader, WandbRunConfig


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


def get_artifact_directory(artifact: wandb.Artifact) -> str:
    dir_paths = set()
    for entry in artifact.manifest.entries.values():
        if entry.ref.startswith("file://"):
            entry_path = Path(entry.ref.replace("file://", ""))
            dir_paths.add(str(entry_path.parent.absolute()))
    match len(dir_paths):
        case 0:
            raise ValueError(
                f"Artifact {artifact.name} does not contain any filesystem references."
            )
        case 1:
            return list(dir_paths)[0]
        case _:
            # TODO: Can this be resolved somehow else???
            dir_string = ",".join(dir_paths)
            raise ValueError(
                f"Artifact {artifact.name} references multiple directories: {dir_string}. "
                "Unable to determine which directory to load."
            )


def resolve_artifact_path(path: str | WandbArtifactConfig, loader: WandbArtifactLoader) -> str:
    """Resolve the actual filesystem path for a path/artifact asset.

    The artifact loader internally handles linking the artifact-to-load to an in-progress run.
    """
    match path:
        case str():
            return path
        case WandbArtifactConfig() as artifact_config:
            artifact = loader.load_artifact(artifact_config)
            return get_artifact_directory(artifact)
        case _:
            raise ValueError(f"Invalid artifact path: {path}")
