import wandb
from pydantic import BaseModel

from lm_buddy.integrations.wandb import WandbArtifactConfig


def copy_pydantic_json(model: BaseModel) -> BaseModel:
    """Copy a Pydantic model through round-trip JSON serialization."""
    return model.__class__.model_validate_json(model.model_dump_json())


class FakeArtifactLoader:
    """Fake implementation of an `ArtifactLoader` with in-memory artifact storage.

    This class bypasses calls to the W&B SDK for using/logging artifacts,
    making it suitable for use in testing when W&B is disabled.

    Note: Artifacts are retrieved from the in-memory storage using just their name,
    not the full W&B path, since the project/entity cannot be inferred when W&B is disabled.
    """

    def __init__(self):
        self._storage: dict[str, wandb.Artifact] = dict()

    def num_artifacts(self) -> int:
        return len(self._storage)

    def get_artifacts(self) -> list[wandb.Artifact]:
        return list(self._storage.values())

    def use_artifact(self, config: WandbArtifactConfig) -> wandb.Artifact:
        return self._storage[config.name]

    def log_artifact(self, artifact: wandb.Artifact) -> None:
        self._storage[artifact.name] = artifact
