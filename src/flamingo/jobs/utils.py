from enum import Enum

from flamingo.integrations.wandb.artifact_config import WandbArtifactConfig
from flamingo.integrations.wandb.utils import get_artifact_directory


class FlamingoJobType(str, Enum):
    """Enumeration of logical job types runnable via the Flamingo."""

    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


def resolve_artifact_path(path_or_artifact: str | WandbArtifactConfig) -> str:
    """Resolve the path to load for an asset.

    If the path is just a string, simply return that.
    If an artifact config, resolve the data path from the artifact's manifest.
    """
    match path_or_artifact:
        case str():
            return path_or_artifact
        case WandbArtifactConfig() as config:
            return get_artifact_directory(config)
        case _:
            raise ValueError(f"Invalid path/artifact: {path_or_artifact}")
