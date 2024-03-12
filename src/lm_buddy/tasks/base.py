import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

from lm_buddy.integrations.wandb import ArtifactLoader, ArtifactType, WandbArtifactConfig


class TaskType(Enum):
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"
    SERVING = "serving"


@dataclass
class TaskResult:
    execution_time: timedelta
    task_type: TaskType
    artifact_type: ArtifactType
    artifact: WandbArtifactConfig


class LMBuddyTask(ABC):
    def __init__(self, artifact_loader: ArtifactLoader):
        self.artifact_loader = artifact_loader

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        pass

    @property
    @abstractmethod
    def artifact_type(self) -> ArtifactType:
        pass

    @abstractmethod
    def _run_internal(self) -> WandbArtifactConfig:
        pass

    def run(self) -> TaskResult:
        start_time = time.time()
        output_artifact = self._run_internal()
        elapsed = time.time() - start_time
        return TaskResult(
            execution_time=timedelta(seconds=elapsed),
            task_type=self.task_type,
            artifact_type=self.artifact_type,
            artifact=output_artifact,
        )
