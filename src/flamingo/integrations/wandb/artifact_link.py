from flamingo.types import BaseFlamingoConfig


class WandbArtifactLink(BaseFlamingoConfig):
    """Data required to retrieve an artifact from W&B."""

    name: str
    version: str = "latest"
    project: str | None = None
    entity: str | None = None

    @property
    def wandb_path(self) -> str:
        """String identifier for the asset on the W&B platform."""
        path = "/".join(x for x in [self.entity, self.project, self.name] if x is not None)
        path = f"{path}:{self.version}"
        return path
