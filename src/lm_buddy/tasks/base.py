import datetime
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, TypeVar

from pydantic.dataclasses import dataclass

from lm_buddy.tasks.configs import LMBuddyTaskConfig
from lm_buddy.tasks.task_output import TaskOutput

ConfigType = TypeVar("ConfigType", bound=LMBuddyTaskConfig)


class TaskType(str, Enum):
    SIMPLE = "simple"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class TaskResult:
    outputs: list[TaskOutput]
    task_type: TaskType
    execution_time: datetime.timedelta


class LMBuddyTask(Generic[ConfigType], ABC):
    """Base interface for a single task within the LMBuddy.

    A task is parameterized by a task configuration containing all of its settings,
    and returns a `TaskResult` referencing the assets produced by the task.
    """

    def __init__(self, config: ConfigType):
        self.config = config

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """Type of task, e.g., 'finetuning' or 'evaluation'."""
        pass

    @abstractmethod
    def _run_internal(self, config: ConfigType) -> list[TaskOutput]:
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
