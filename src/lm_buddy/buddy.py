from lm_buddy.integrations.wandb import ArtifactLoader, WandbArtifactLoader
from lm_buddy.jobs.configs.finetuning import FinetuningJobConfig


class LMBuddy:
    def __init__(self, artifact_loader: ArtifactLoader = WandbArtifactLoader()):
        self._artifact_loader = artifact_loader

    def finetune(self, config: FinetuningJobConfig) -> None:
        pass
