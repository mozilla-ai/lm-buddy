from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urlparse

import wandb


class ArtifactType(str, Enum):
    """Enumeration of artifact types used by the LM Buddy."""

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
    artifact_name: str,
    artifact_type: ArtifactType,
    dir_path: str | Path,
    *,
    reference: bool = False,
    entry_name: str | None = None,
    max_objects: int | None = None,
) -> wandb.Artifact:
    """Build an artifact containing the contents of a directory.

    Args:
        artifact_name (str): Name of the artifact.
        artifact_type (ArtifactType): Type of artifact.
        dir_path (str | Path): Directory path to include in the artifact.

    Keyword Args:
        reference (bool): Only reference the directory, do not copy contents. Defaults to False.
        entry_name (str | None): Name for directory within the artifact. Defaults to None.
        max_objects (int | None): Max objects to include in the artifact. Defaults to None.

    Returns:
        wandb.Artifact: The generated artifact.
    """
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
    artifact_name: str,
    artifact_type: ArtifactType,
    columns: list[str],
    tables: dict[str, list[list[Any]]],
) -> wandb.Artifact:
    """Build an artifact containing one or more table entries.

    Args:
        artifact_name (str): Name of the artifact.
        artifact_type (ArtifactType): Type of artifact.
        columns (list[str]): Column names for the tables.
        tables (dict[str, list[list[Any]]]): Mapping from table name to table rows.

    Returns:
        wandb.Artifact: The artifact containing the table(s).
    """
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    for table_name, table_data in tables.items():
        table = wandb.Table(data=table_data, columns=columns)
        artifact.add(table, name=table_name)
    return artifact
