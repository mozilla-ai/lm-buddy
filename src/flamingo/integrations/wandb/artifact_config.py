import wandb

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

    def load_artifact(self, link: WandbArtifactConfig) -> wandb.Artifact:
        if self._run is not None:
            # Retrieves the artifact and links it as an input to the run
            return self._run.use_artifact(link.wandb_path)
        else:
            # Retrieves the artifact outside of the run
            api = wandb.Api()
            return api.artifact(link.wandb_path)
