from typing import Protocol

import wandb

from lm_buddy.integrations.wandb import WandbArtifactConfig


class ArtifactLoader(Protocol):
    """Base interface for using/logging artifacts.

    Note: If/when we decide to support other tracking services (e.g., MLFlow, CometML),
    this interface should be abstracted to handle the types for those respective services.
    """

    def use_artifact(self, config: WandbArtifactConfig) -> wandb.Artifact:
        """Load an artifact from its configuration.

        If a W&B run is active, the artifact is declared as an input to the run.
        If not, the artifact is retrieved outside of the run.
        """
        pass

    def log_artifact(self, artifact: wandb.Artifact) -> None:
        """Log an artifact, declaring it as an output of the currently active W&B run."""
        pass


class WandbArtifactLoader:
    """Weights & Biases implementation of the `ArtifactLoader` protocol.

    This class makes external calls to the W&B API and is not suitable for test environments.
    """

    def use_artifact(self, config: WandbArtifactConfig) -> wandb.Artifact:
        if wandb.run is not None:
            # Retrieves the artifact and links it as an input to the run
            return wandb.use_artifact(config.wandb_path())
        else:
            # Retrieves the artifact outside of the run
            api = wandb.Api()
            return api.artifact(config.wandb_path())

    def log_artifact(self, artifact: wandb.Artifact) -> None:
        return wandb.log_artifact(artifact)
