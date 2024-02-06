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


def get_artifact_filesystem_path(
    artifact: wandb.Artifact,
    *,
    download_root_path: str | None = None,
) -> Path:
    """Get the directory containing the artifact's data.

    If the artifact references data already on the filesystem, simply return that path.
    If not, downloads the artifact (with the specified `download_root_path`)
    and returns the newly created artifact directory path.
    """
    for entry in artifact.manifest.entries.values():
        match urlparse(entry.ref):
            case ParseResult(scheme="file", path=file_path):
                return Path(file_path).parent
    # No filesystem references found in the manifest -> download the artifact
    download_path = artifact.download(root=download_root_path)
    return Path(download_path)


def build_directory_artifact(
    dir_path: str | Path,
    artifact_name: str,
    artifact_type: ArtifactType,
    *,
    reference: bool = False,
    entry_name: str | None = None,
    max_objects: int | None = None,
) -> wandb.Artifact:
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    if reference:
        # Right now, we are assuming a fixed "file" URI scheme
        # We can pass the URI scheme if necessary later
        artifact.add_reference(
            uri=f"{ArtifactURIScheme.FILE}://{dir_path}",
            max_objects=max_objects,
            name=entry_name,
        )
    else:
        artifact.add_dir(str(dir_path), name=entry_name)
    return artifact


def build_table_artifact(
    table_data: dict[str, dict[str, float]],
    artifact_name: str,
    artifact_type: ArtifactType,
) -> wandb.Artifact:
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    for table_name, data in table_data.items():
        table = wandb.Table(data=data, columns=["metric", "value"])
        artifact.add(table, name=f"{table_name}-table")
    return artifact


def log_directory_contents(
    dir_path: str | Path,
    artifact_name: str,
    artifact_type: ArtifactType,
    *,
    entry_name: str | None = None,
) -> wandb.Artifact:
    """Log the contents of a directory as an artifact of the active run.

    A run should already be initialized before calling this method.
    If not, an exception will be thrown.

    Args:
        dir_path (str | Path): Path to the artifact directory.
        artifact_name (str): Name of the artifact.
        artifact_type (ArtifactType): Type of the artifact to create.
        entry_name (str, optional): Name within the artifact to add the directory contents.

    Returns:
        The `wandb.Artifact` that was produced

    """
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_dir(str(dir_path), name=entry_name)
    return wandb.log_artifact(artifact)


def log_directory_reference(
    dir_path: str | Path,
    artifact_name: str,
    artifact_type: ArtifactType,
    *,
    scheme: ArtifactURIScheme = ArtifactURIScheme.FILE,
    entry_name: str | None = None,
    max_objects: int | None = None,
) -> wandb.Artifact:
    """Log a reference to a directory's contents as an artifact of the active run.

    A run should already be initialized before calling this method.
    If not, an exception will be thrown.

    Args:
        dir_path (str | Path): Path to the artifact directory.
        artifact_name (str): Name of the artifact.
        artifact_type (ArtifactType): Type of the artifact to create.
        scheme (ArtifactURIScheme): URI scheme to prepend to the artifact path.
            Defaults to `ArtifactURIScheme.FILE` for filesystem references.
        entry_name (str, optional): Name within the artifact to add the directory reference.
        max_objects (int, optional): Max number of objects allowed in the artifact.

    Returns:
        The `wandb.Artifact` that was produced

    """
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
    artifact.add_reference(
        uri=f"{scheme}://{dir_path}",
        name=entry_name,
        max_objects=max_objects,
    )
    return wandb.log_artifact(artifact)


class WandbArtifactLoader:
    """Abstraction around using and logging W&B artifacts."""

    def load_artifact(self, config: WandbArtifactConfig) -> wandb.Artifact:
        """Load an artifact from the artifact config.

        If a W&B run is active, the artifact is used as an input to the run.
        If not, the artifact is pulled from the W&B API outside of the run.
        """
        if wandb.run is not None:
            # Retrieves the artifact and links it as an input to the run
            return wandb.use_artifact(config.wandb_path())
        else:
            # Retrieves the artifact outside of the run
            api = wandb.Api()
            return api.artifact(config.wandb_path())

    def log_artifact(self, artifact: wandb.Artifact) -> None:
        """Log an artifact as an output of the currently active W&B run.

        An exception is raised if a run is not active.
        """
        return wandb.log_artifact(artifact)
