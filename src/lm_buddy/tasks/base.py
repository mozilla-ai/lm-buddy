import datetime
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lm_buddy.integrations.wandb import ArtifactLoader, ArtifactType


class TaskType(Enum):
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"
    SERVING = "serving"


@dataclass
class TaskResult:
    task_type: TaskType
    artifact_type: ArtifactType
    artifact_path: Path
    execution_time: datetime.timedelta


class BaseLMBuddyTask(ABC):
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
    def _run_internal(self) -> Path:
        pass

    def run(self) -> TaskResult:
        start_time = time.time()
        artifact_path = self._run_internal()
        elapsed = time.time() - start_time
        return TaskResult(
            task_type=self.task_type,
            artifact_type=self.artifact_type,
            artifact_path=artifact_path,
            execution_time=datetime.timedelta(seconds=elapsed),
        )
