import contextlib
from typing import Any

import wandb
from wandb.apis.public import Run as ApiRun

from flamingo.integrations.wandb import ArtifactType, WandbArtifactConfig, WandbRunConfig


@contextlib.contextmanager
def wandb_init_from_config(config: WandbRunConfig, *, resume: str | None = None):
    """Initialize a W&B run from the internal run configuration."""
    init_kwargs = dict(
        id=config.run_id,
        name=config.name,
        project=config.project,
        entity=config.entity,
        group=config.run_group,
    )
    with wandb.init(**init_kwargs, resume=resume) as run:
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


def get_wandb_artifact(config: WandbArtifactConfig) -> wandb.Artifact:
    """Load an artifact from the provided config.

    If a W&B run is active, the artifact is loaded via the run as an input.
    If not, the artifact is pulled from the W&B API outside of the run.
    """
    if wandb.run is not None:
        # Retrieves the artifact and links it as an input to the run
        return wandb.run.use_artifact(config.wandb_path())
    else:
        # Retrieves the artifact outside of the run
        api = wandb.Api()
        return api.artifact(config.wandb_path())


def resolve_artifact_path(path: str | WandbArtifactConfig) -> str:
    """Resolve the actual filesystem path from an artifact/path reference.

    If the provided path is just a string, return the value directly.
    If an artifact, download it from W&B (and link it to an in-progress run)
    to retrieve the actual data directory.
    """
    match path:
        case str():
            return path
        case WandbArtifactConfig() as config:
            artifact = get_wandb_artifact(config)
            return artifact.download()
        case _:
            raise ValueError(f"Invalid artifact path: {path}")


def default_artifact_name(name: str, artifact_type: ArtifactType) -> str:
    """A default name for an artifact based on the run name and type."""
    return f"{name}-{artifact_type}"
