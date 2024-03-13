from lm_buddy.integrations.wandb.artifact_loader import ArtifactLoader
from lm_buddy.paths import LoadableAssetPath
from lm_buddy.tasks.common import LMBuddyTask


class SimpleTask(LMBuddyTask):
    def _run_internal(self, artifact_loader: ArtifactLoader) -> LoadableAssetPath:
        pass
