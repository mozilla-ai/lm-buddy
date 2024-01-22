from enum import Enum
from pathlib import Path
from urllib.parse import ParseResult, urlparse

import wandb

from flamingo.integrations.wandb import WandbArtifactConfig


class ArtifactType(str, Enum):
    """Enumeration of artifact types used by the Flamingo."""

    DATASET = "dataset"
    MODEL = "model"
    TOKENIZER = "tokenizer"
    EVALUATION = "evaluation"


class ArtifactURIScheme(str, Enum):
    """Enumeration of URI schemes to use in a reference artifact."""

    FILE = "file"
    HTTP = "http"
    HTTPS = "https"
    S3 = "s3"
    GCS = "gs"


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


def get_artifact_path(
    config: WandbArtifactConfig,
    *,
    download_root: str | None = None,
) -> str:
    """Get the directory containing the artifact's data.

    If the artifact references data already on the filesystem, simply return that path.
    If not, downloads the artifact (with the specified `download_root`)
    and returns the newly created artifact directory path.
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
    uri_scheme: ArtifactURIScheme | None = None,
    max_objects: int | None = None,
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
        uri_scheme (ArtifactURIScheme, optional): URI scheme to prepend to the artifact path.
            When provided, the artifact is logged as a reference to this path.
        max_objects (int, optional): Max number of objects allowed in the artifact.
            Only used when creating reference artifacts.

    Returns:
        The `wandb.Artifact` that was logged

    """
    artifact = wandb.Artifact(name=name, type=artifact_type)
    if uri_scheme is not None:
        artifact.add_reference(f"{uri_scheme}://{path}", max_objects=max_objects)
    else:
        artifact.add_dir(str(path))
    # Log artifact to the currently active run
    return wandb.run.log_artifact(artifact)
