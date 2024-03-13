from lm_buddy.integrations.wandb import ArtifactLoader
from lm_buddy.tasks import TaskOutput
from lm_buddy.tasks.base import LMBuddyTask, TaskType
from lm_buddy.tasks.configs.simple import SimpleTaskConfig


class SimpleTask(LMBuddyTask[SimpleTaskConfig]):
    """Simple implementation to demonstrate the task interface."""

    def __init__(self, config: SimpleTaskConfig, artifact_loader: ArtifactLoader):
        super().__init__(self, config, TaskType.SIMPLE, artifact_loader)

    def _run_internal(self) -> list[TaskOutput]:
        print(f"The magic number is {self.config.magic_number}")
        return []
