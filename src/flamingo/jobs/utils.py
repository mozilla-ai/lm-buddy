from enum import Enum

from flamingo.integrations.wandb import WandbArtifactConfig, get_artifact_filesystem_path


class FlamingoJobType(str, Enum):
    """Enumeration of logical job types runnable via the Flamingo."""

    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


def resolve_artifact_load_path(artifact_or_path: str | WandbArtifactConfig) -> str:
    """Resolve the path to load for an asset.

    If the path is just a string, simply return that.
    If an artifact config, resolve the data path from the artifact's manifest.
    """
    match artifact_or_path:
        case str():
            return artifact_or_path
        case WandbArtifactConfig() as config:
            return get_artifact_filesystem_path(config)
        case _:
            raise ValueError(f"Invalid artifact/path: {artifact_or_path}")
