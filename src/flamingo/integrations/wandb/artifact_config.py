import re

from flamingo.types import BaseFlamingoConfig


class WandbArtifactConfig(BaseFlamingoConfig):
    """Configuration required to retrieve an artifact from W&B."""

    name: str
    project: str
    version: str = "latest"
    entity: str | None = None

    @classmethod
    def from_wandb_path(cls, path: str) -> "WandbArtifactConfig":
        """Construct a configuration from the full W&B name."""
        match = re.search(r"((.*)\/)?(.*)\/(.*)\:(.*)", path)
        if match is not None:
            entity, project, name, version = match.groups()[1:]
            return cls(name=name, project=project, version=version, entity=entity)
        raise ValueError(f"Invalid artifact path: {path}")

    def wandb_path(self) -> str:
        """String identifier for the asset on the W&B platform."""
        path = f"{self.project}/{self.name}:{self.version}"
        if self.entity is not None:
            path = f"{self.entity}/{path}"
        return path
