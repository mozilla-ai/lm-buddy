from pathlib import Path
from typing import Any

import wandb
from wandb.apis.public import Run as ApiRun

from flamingo.integrations.wandb import WandbArtifactConfig, WandbRunConfig


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


def get_artifact_directory(artifact: wandb.Artifact) -> Path:
    entry_paths = [
        e.replace("file://", "")
        for e in artifact.manifest.entries.values()
        if e.startswith("file://")
    ]
    dir_paths = {Path(e).parent.absolute() for e in entry_paths}
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


class WandbArtifactLoader:
    """Helper class for loading W&B artifacts and linking them to runs."""

    def __init__(self, run: wandb.run):
        self._run = run

    def load_artifact(self, config: WandbArtifactConfig) -> wandb.Artifact:
        """Load an artifact from the provided config.

        If a a W&B run is available, the artifact is loaded via the run as an input.
        If not, the artifact is pulled from the W&B API outside of the run.
        """
        if self._run is not None:
            # Retrieves the artifact and links it as an input to the run
            return self._run.use_artifact(config.wandb_path())
        else:
            # Retrieves the artifact outside of the run
            api = wandb.Api()
            return api.artifact(config.wandb_path())

    def resolve_artifact_path(self, path: str | WandbArtifactConfig) -> str:
        """Resolve the actual filesystem path for an artifact.

        If the path is just a string, returns the value directly.
        If an artifact, the loader will load it from W&B (and link it to an in-progress run)
        and resolve the filesystem path from the artifact manifest.
        """
        match path:
            case str():
                return path
            case WandbArtifactConfig() as artifact_config:
                artifact = self.load_artifact(artifact_config)
                artifact_path = get_artifact_directory(artifact)
                return str(artifact_path)
            case _:
                raise ValueError(f"Invalid artifact path: {path}")
