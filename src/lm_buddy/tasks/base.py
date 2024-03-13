import datetime
import time
from abc import ABC, abstractmethod
from enum import Enum

from pydantic import dataclass

from lm_buddy.integrations.wandb import ArtifactLoader
from lm_buddy.tasks.task_output import TaskOutput


class TaskType(str, Enum):
    SIMPLE = "simple"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class TaskResult:
    task_type: TaskType
    outputs: list[TaskOutput]
    execution_time: datetime.timedelta


class LMBuddyTask(ABC):
    """Base interface for a single task within the LMBuddy.

    A task is parameterized by a task configuration containing all of its settings,
    and returns a `TaskResult` referencing the assets produced by the task.
    """

    def __init__(self, task_type: TaskType, artifact_loader: ArtifactLoader):
        self.task_type = task_type
        self.artifact_loader = artifact_loader

    @abstractmethod
    def _run_internal(self) -> list[TaskOutput]:
        pass

    def run(self) -> TaskResult:
        start_time = time.time()
        outputs = self._run_internal()
        elapsed = time.time() - start_time
        return TaskResult(
            outputs=outputs,
            task_type=self.task_type,
            execution_time=datetime.timedelta(seconds=elapsed),
        )
