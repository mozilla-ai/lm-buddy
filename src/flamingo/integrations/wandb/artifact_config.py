import wandb

from flamingo.integrations.wandb.utils import get_artifact_directory
from flamingo.types import BaseFlamingoConfig


class WandbArtifactConfig(BaseFlamingoConfig):
    """Configuration required to retrieve an artifact from W&B."""

    name: str
    version: str = "latest"
    project: str | None = None
    entity: str | None = None

    def wandb_path(self) -> str:
        """String identifier for the asset on the W&B platform."""
        path = "/".join(x for x in [self.entity, self.project, self.name] if x is not None)
        path = f"{path}:{self.version}"
        return path


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

    def resolve_path_reference(self, path: str | WandbArtifactConfig) -> str:
        """Resolve the actual filesystem path from an artifact/path reference.

        If the provided path is just a string, return the value directly.
        If an artifact, load it from W&B (and link it to an in-progress run)
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
