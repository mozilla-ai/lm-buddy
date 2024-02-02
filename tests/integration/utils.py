from pathlib import Path

import wandb

from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.integrations.wandb.artifact_logger import WandbArtifactLogger


class FakeWandbArtifactLogger(WandbArtifactLogger):
    """Helper that can be preloaded with artifacts for offline retrieval in tests."""

    def __init__(self):
        self._registry = dict()

    # Test helpers
    def _set_artifact(self, path: str, artifact: wandb.Artifact) -> None:
        self.registry[path] = artifact

    # Overloads
    def get_artifact(self, config: WandbArtifactConfig) -> wandb.Artifact:
        return self.registry[config.wandb_path()]

    def get_artifact_filesystem_path(
        self,
        config: WandbArtifactConfig,
        *,
        download_root_path: str | None = None,
    ) -> Path:
        return super().get_artifact_filesystem_path(config, download_root_path=download_root_path)
