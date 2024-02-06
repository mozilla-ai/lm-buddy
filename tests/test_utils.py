import wandb
from pydantic import BaseModel

from flamingo.integrations.wandb import WandbArtifactConfig, WandbArtifactLoader


def copy_pydantic_json(model: BaseModel) -> BaseModel:
    """Copy a Pydantic model through round-trip JSON serialization."""
    return model.__class__.model_validate_json(model.model_dump_json())


class FakeWandbArtifactLoader(WandbArtifactLoader):
    def __init__(self):
        self._storage = dict()

    def use_artifact(self, config: WandbArtifactConfig) -> wandb.Artifact:
        return self._storage[config.name]

    def log_artifact(self, artifact: wandb.Artifact) -> None:
        self._storage[artifact.name] = artifact
