import datetime
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from lm_buddy.integrations.huggingface import LoadableAssetPath
from lm_buddy.integrations.wandb import ArtifactLoader


class TaskType(str, Enum):
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class TaskResult:
    task_type: TaskType
    asset_paths: list[LoadableAssetPath]
    execution_time: datetime.timedelta


class LMBuddyTask(ABC):
    """Base interface for a single task within the LMBuddy.

    A task is parameterized by a task configuration containing all of its settings,
    and returns a `TaskResult` referencing the assets produced by the task.
    """

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        pass

    @abstractmethod
    def _run_internal(self, artifact_loader: ArtifactLoader) -> list[LoadableAssetPath]:
        pass

    def run(self, artifact_loader: ArtifactLoader) -> TaskResult:
        start_time = time.time()
        asset_paths = self._run_internal(artifact_loader)
        elapsed = time.time() - start_time
        return TaskResult(
            task_type=self.task_type,
            asset_paths=asset_paths,
            execution_time=datetime.timedelta(seconds=elapsed),
        )
