from pathlib import Path
from urllib.parse import ParseResult, urlparse

import wandb

from flamingo.integrations.wandb import ArtifactType, ArtifactURIScheme, WandbArtifactConfig


class WandbArtifactLogger:
    """Collection of utilities for retrieving/logging W&B artifacts.

    These methods are placed on a class so they can be injected into job code.
    """

    def get_artifact(self, config: WandbArtifactConfig) -> wandb.Artifact:
        """Load an artifact from the artifact config.

        If a W&B run is active, the artifact is loaded via the run as an input.
        If not, the artifact is pulled from the W&B API outside of the run.
        """
        if wandb.run is not None:
            # Retrieves the artifact and links it as an input to the run
            return wandb.use_artifact(config.wandb_path())
        else:
            # Retrieves the artifact outside of the run
            api = wandb.Api()
            return api.artifact(config.wandb_path())

    def get_artifact_filesystem_path(
        self,
        config: WandbArtifactConfig,
        *,
        download_root_path: str | None = None,
    ) -> Path:
        """Get the directory containing the artifact's data.

        If the artifact references data already on the filesystem, simply return that path.
        If not, downloads the artifact (with the specified `download_root_path`)
        and returns the newly created artifact directory path.
        """
        artifact = self.get_artifact(config)
        for entry in artifact.manifest.entries.values():
            match urlparse(entry.ref):
                case ParseResult(scheme="file", path=file_path):
                    return Path(file_path).parent
        # No filesystem references found in the manifest -> download the artifact
        download_path = artifact.download(root=download_root_path)
        return Path(download_path)

    def log_directory_contents(
        self,
        dir_path: str | Path,
        artifact_name: str,
        artifact_type: ArtifactType,
        *,
        entry_name: str | None = None,
    ) -> wandb.Artifact:
        """Log the contents of a directory as an artifact of the active run.

        A run should already be initialized before calling this method.
        If not, an exception will be thrown.

        Args:
            dir_path (str | Path): Path to the artifact directory.
            artifact_name (str): Name of the artifact.
            artifact_type (ArtifactType): Type of the artifact to create.
            entry_name (str, optional): Name within the artifact to add the directory contents.

        Returns:
            The `wandb.Artifact` that was produced

        """
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_dir(str(dir_path), name=entry_name)
        return wandb.log_artifact(artifact)

    def log_directory_reference(
        self,
        dir_path: str | Path,
        artifact_name: str,
        artifact_type: ArtifactType,
        *,
        scheme: ArtifactURIScheme = ArtifactURIScheme.FILE,
        entry_name: str | None = None,
        max_objects: int | None = None,
    ) -> wandb.Artifact:
        """Log a reference to a directory's contents as an artifact of the active run.

        A run should already be initialized before calling this method.
        If not, an exception will be thrown.

        Args:
            dir_path (str | Path): Path to the artifact directory.
            artifact_name (str): Name of the artifact.
            artifact_type (ArtifactType): Type of the artifact to create.
            scheme (ArtifactURIScheme): URI scheme to prepend to the artifact path.
                Defaults to `ArtifactURIScheme.FILE` for filesystem references.
            entry_name (str, optional): Name within the artifact to add the directory reference.
            max_objects (int, optional): Max number of objects allowed in the artifact.

        Returns:
            The `wandb.Artifact` that was produced

        """
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
        artifact.add_reference(
            uri=f"{scheme}://{dir_path}",
            name=entry_name,
            max_objects=max_objects,
        )
        return wandb.log_artifact(artifact)
