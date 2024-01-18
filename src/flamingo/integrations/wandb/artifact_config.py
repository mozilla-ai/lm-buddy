from pathlib import Path

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

    def resolve_artifact_path(self, path: str | WandbArtifactConfig) -> str:
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
                artifact_path = self._extract_base_path(artifact)
                return str(artifact_path)
            case _:
                raise ValueError(f"Invalid artifact path: {path}")

    def _extract_base_path(self, artifact: wandb.Artifact) -> Path:
        """Extract the base filesystem path from entries in an artifact.

        An error is raised if the artifact contains ether zero or more than one references
        to distinct filesystem directories.
        """
        entry_paths = [
            e.ref.replace("file://", "")
            for e in artifact.manifest.entries.values()
            if e.ref.startswith("file://")
        ]
        dir_paths = {Path(e).parent.absolute() for e in entry_paths}
        match len(dir_paths):
            case 0:
                raise ValueError(
                    f"Artifact {artifact.name} does not contain any filesystem references."
                )
            case 1:
                return list(dir_paths)[0]
            case _:
                # TODO: Can this be resolved somehow else???
                dir_string = ",".join(dir_paths)
                raise ValueError(
                    f"Artifact {artifact.name} references multiple directories: {dir_string}. "
                    "Unable to determine which directory to load."
                )
