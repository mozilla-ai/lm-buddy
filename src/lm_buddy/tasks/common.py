import datetime
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from lm_buddy.integrations.huggingface import HuggingFaceAssetPath
from lm_buddy.integrations.wandb import ArtifactLoader, ArtifactType
from lm_buddy.types import BaseLMBuddyConfig


class TaskType(Enum):
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class TaskResult:
    task_type: TaskType
    artifact_type: ArtifactType
    artifact_path: HuggingFaceAssetPath
    execution_time: datetime.timedelta


class LMBuddyTask(ABC):
    """Abstract interface for a single task within the LMBuddy.

    A task is parameterized by a single configuration containing all of its settings,
    and returns a `TaskResult` referencing the artifact produced by the task.
    """

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        pass

    @property
    @abstractmethod
    def task_config(self) -> BaseLMBuddyConfig:  # TODO: Specialize this type
        pass

    @property
    @abstractmethod
    def artifact_type(self) -> ArtifactType:
        pass

    @abstractmethod
    def _run_internal(self, artifact_loader: ArtifactLoader) -> HuggingFaceAssetPath:
        pass

    def run(self, artifact_loader: ArtifactLoader) -> TaskResult:
        start_time = time.time()
        artifact_path = self._run_internal(artifact_loader)
        elapsed = time.time() - start_time
        return TaskResult(
            task_type=self.task_type,
            artifact_type=self.artifact_type,
            artifact_path=artifact_path,
            execution_time=datetime.timedelta(seconds=elapsed),
        )
