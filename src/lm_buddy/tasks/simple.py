from lm_buddy.integrations.wandb import ArtifactLoader
from lm_buddy.paths import LoadableAssetPath
from lm_buddy.tasks.base import LMBuddyTask


class SimpleTask(LMBuddyTask):
    """Simple implementation to demonstrate the task interface."""

    def __init__(self, artifact_loader: ArtifactLoader):
        super().__init__(self, artifact_loader)

    def _run_internal(self) -> LoadableAssetPath:
        pass
