from dataclasses import dataclass
from typing import Any

from lm_buddy.integrations.huggingface import AutoModelConfig
from lm_buddy.integrations.wandb import (
    ArtifactLoader,
    WandbArtifactLoader,
    WandbRunConfig,
)
from lm_buddy.tasks.base import TaskResult


@dataclass
class RunResult:
    execution_time: float
    task_results: list[TaskResult]


class LMBuddy:
    """Your buddy in the (L)LM space.

    The `LMBuddy` orchestrates a simple finetuning and evaluation workflow.
    The workflow begins with a base model, and the following tasks can be added:
    - A finetuning task to generate a tuned model
    - A series of evaluation tasks
    - A serving task

    The workflow is executed in order of finetuning -> evaluation -> serving.
    If finetuning is not specified, evaluation and serving utilizes the base model.
    """

    def __init__(
        self,
        *,
        model_config: AutoModelConfig,
        tracking_config: WandbRunConfig | None = None,
        artifact_loader: ArtifactLoader = WandbArtifactLoader(),
    ):
        self.model_config = model_config
        self.tracking_config = tracking_config
        self.artifact_loader = artifact_loader

        self.finetuning_config: "FinetuningTaskConfig" | None = None
        self.evaluation_configs: dict[str, "EvaluationTaskConfig"] = []

    def add_finetuning_task(self, config: "FinetuningTaskConfig") -> "LMBuddy":
        self.finetuning_task = config
        return self

    def add_evaluation_task(self, config: "EvaluationTaskConfig") -> "LMBuddy":
        self.evaluation_tasks.append(config)
        return self

    def add_serving_task(self, config: Any) -> "LMBuddy":
        raise NotImplementedError("Serving is not yet implemented in lm-buddy.")

    def run(self) -> RunResult:
        task_results = []

        final_model = self.input_model
        if self.finetuning_task is not None:
            finetuning_result = self.finetuning_task.run()
            final_model = ...  # Run finetuning

        eval_results = []
        for task in self.evaluation_tasks:
            task_result = task.run()  # Run eval using final model and task config
            eval_results.append(task_result)

        total_execution_time = sum(x.execution_time for x in task_results)
        return RunResult(
            final_model=final_model,
            evaluation_results=eval_results,
            execution_time=total_execution_time,
        )
