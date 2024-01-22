import contextlib
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urlparse

import wandb
from wandb.apis.public import Run as ApiRun

from flamingo.integrations.wandb import ArtifactType, WandbArtifactConfig, WandbRunConfig
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


def default_artifact_name(name: str, artifact_type: ArtifactType) -> str:
    """A default name for an artifact based on the run name and type."""
    return f"{name}-{artifact_type}"


def get_wandb_artifact(config: WandbArtifactConfig) -> wandb.Artifact:
    """Load an artifact from the artifact config.

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


def get_artifact_directory(
    config: WandbArtifactConfig,
    *,
    download_root: str | None = None,
) -> str:
    """Get the directory containing the artifact's data.

    If the artifact references data already on the filesystem, simply returns that path.
    If not, downloads the artifact (with the specified `download_root`)
    and returns the newly created artifact directory.
    """
    artifact = get_wandb_artifact(config)
    for entry in artifact.manifest.entries.values():
        match urlparse(entry.ref):
            case ParseResult(scheme="file", path=file_path):
                return str(Path(file_path).parent)
    # No filesystem references found in the manifest -> download the artifact
    return artifact.download(root=download_root)


def log_artifact_from_path(
    name: str,
    path: str | Path,
    artifact_type: ArtifactType,
    *,
    reference_scheme: str | None = None,
) -> wandb.Artifact:
    """Log an artifact containing the contents of a directory to the currently active run.

    A run should already be initialized before calling this method.
    If not, an exception will be thrown.

    Example usage:
    ```
    with wandb_init_from_config(run_config):
        log_artifact_from_path(...)
    ```

    Args:
        name (str): Name of the artifact
        path (str | Path): Path to the artifact directory
        artifact_type (ArtifactType): Type of the artifact to create
        reference_scheme (str, optional): URL scheme to prepend to the artifact path.
            When provided, the artifact is logged as a reference to this path.

    Returns:
        The `wandb.Artifact` that was logged

    """
    artifact = wandb.Artifact(name=name, type=artifact_type)
    if reference_scheme is not None:
        artifact.add_reference(f"{reference_scheme}://{path}")
    else:
        artifact.add_dir(str(path))
    # Log artifact to the currently active run
    return wandb.run.log_artifact(artifact)
