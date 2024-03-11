from dataclasses import dataclass
from time import time

from lm_buddy.integrations.huggingface import AutoModelConfig
from lm_buddy.integrations.wandb import WandbRunConfig


@dataclass
class LMBuddyRunResult:
    final_model: AutoModelConfig | None
    evaluation_results: list
    execution_time: float


class LMBuddy:
    """Your buddy in the (L)LM space.

    The `LMBuddy` orchestrates a simple finetuning and evaluation workflow.
    The workflow begins with a base model and can be augmented with the following tasks:
    - A finetuning task to generate a tuned model
    - A series of evaluation tasks
    - A serving task

    The workflow is executed in order of finetuning -> evaluation -> serving.
    If finetuning is not specified, evaluation and serving utilizes the base model.
    """

    def __init__(self, base_model: AutoModelConfig, tracking: WandbRunConfig | None = None):
        self.tracking = tracking
        self.base_model = base_model

        self.finetuning_task: "FinetuningTaskConfig" | None = None
        self.evaluation_tasks: list["EvaluationTaskConfig"] = []
        self.serving_task: "ServingTaskConfig" | None = None

    def add_finetuning(self, config: "FinetuningTaskConfig") -> "LMBuddy":
        self.finetuning_task = config
        return self

    def add_evaluation(self, config: "EvaluationTaskConfig") -> "LMBuddy":
        self.evaluation_tasks.append(config)
        return self

    def add_serving(self, config: "ServingTaskConfig") -> "LMBuddy":
        self.serving_task = config
        return self

    def run(self) -> LMBuddyRunResult:
        start_time = time.time()

        final_model = self.base_model
        if self.finetuning_task is not None:
            final_model = ...  # Run finetuning

        eval_results = []
        for eval_task in self.evaluation_tasks:
            result = ...  # Run eval using final model and task config
            eval_results.append(result)

        if self.serving_task is not None:
            # Serve the final model
            pass

        execution_time = time.time() - start_time
        return LMBuddyRunResult(
            final_model=final_model,
            evaluation_results=eval_results,
            execution_time=execution_time,
        )
