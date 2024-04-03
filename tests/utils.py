import wandb
from pydantic import BaseModel
from wandb.sdk.artifacts.artifact_state import ArtifactState


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

    def __init__(self, project: str = "test-project", entity: str = "test-entity"):
        self.project = project
        self.entity = entity
        self._storage: dict[str, wandb.Artifact] = dict()

    def _set_commit_attributes(self, artifact: wandb.Artifact) -> wandb.Artifact:
        artifact._project = self.project
        artifact._entity = self.entity
        artifact._version = "latest"
        artifact._state = ArtifactState.COMMITTED

        # W&B does this after logging an artifact
        name_has_version = len(artifact.name.split(":")) > 1
        if not name_has_version:
            artifact._name = f"{artifact._name}:{artifact._version}"

        return artifact

    def num_artifacts(self) -> int:
        return len(self._storage)

    def get_artifacts(self) -> list[wandb.Artifact]:
        return list(self._storage.values())

    def use_artifact(self, artifact_path: str) -> wandb.Artifact:
        return self._storage[artifact_path]

    def log_artifact(self, artifact: wandb.Artifact) -> wandb.Artifact:
        """Store the artifact in-memory and update its attributes to mimic the W&B platform."""
        artifact = self._set_commit_attributes(artifact)
        self._storage[artifact.qualified_name] = artifact
        return artifact
