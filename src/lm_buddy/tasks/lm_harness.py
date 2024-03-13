from typing import Any

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.openai_completions import OpenaiCompletionsLM

from lm_buddy.integrations.huggingface import (
    AutoModelConfig,
    HuggingFaceAssetLoader,
    LoadableAssetPath,
    resolve_peft_and_pretrained,
)
from lm_buddy.integrations.wandb import (
    ArtifactLoader,
    ArtifactType,
    WandbResumeMode,
    build_table_artifact,
    default_artifact_name,
    wandb_init_from_config,
)
from lm_buddy.tasks.base import LMBuddyTask, TaskType
from lm_buddy.tasks.configs import LMHarnessTaskConfig, LocalChatCompletionsConfig
from lm_buddy.tasks.task_output import TaskOutput


def get_numeric_metrics(
    results: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, float]]]:
    """Filter non-numeric values from the evaluation results.

    This is necessary because artifact tables must have a single datatype for each column.

    lm-harness returns mostly numeric values, but there are also some misc string-valued metrics.
    Filtering down to only numeric values allows us to produce a valid table artifact.
    """
    numeric_results = {}
    for key, data in results.items():
        numeric_rows = [(k, v) for k, v in data.items() if isinstance(v, int | float)]
        numeric_results[key] = numeric_rows
    return numeric_results


class LMHarnessTask(LMBuddyTask[LMHarnessTaskConfig]):
    """Evaluation task with lm-evaluation-harness framework."""

    def __init__(self, config: LMHarnessTaskConfig, artifact_loader: ArtifactLoader):
        super().__init__(self, config, TaskType.EVALUATION, artifact_loader)

    def _run_internal(self) -> list[TaskOutput]:
        print(
            f"Running lm-harness task with configuration:\n{self.config.model_dump_json(indent=2)}"
        )

        if self.config.tracking is not None:
            with wandb_init_from_config(
                self.config.tracking,
                parameters=self.config.evaluator,  # Log eval settings in W&B run
                resume=WandbResumeMode.ALLOW,
                job_type=self.task_type,
            ) as run:
                eval_results = self._load_and_evaluate(self.config, self.artifact_loader)
                eval_artifact = build_table_artifact(
                    artifact_name=default_artifact_name(run.name, ArtifactType.EVALUATION),
                    artifact_type=ArtifactType.EVALUATION,
                    columns=["metric", "value"],
                    tables=eval_results,
                )
                print("Logging artifact for evaluation results...")
                self.artifact_loader.log_artifact(eval_artifact)
        else:
            self._load_and_evaluate()

    def _load_and_evaluate(self) -> dict[str, list[tuple[str, float]]]:
        print("Initializing lm-harness tasks...")
        lm_eval.tasks.initialize_tasks()

        llm = self._load_harness_model()
        eval_results = lm_eval.simple_evaluate(
            model=llm,
            tasks=self.config.evaluator.tasks,
            batch_size=self.config.evaluator.batch_size,
            num_fewshot=self.config.evaluator.num_fewshot,
            limit=self.config.evaluator.limit,
            log_samples=False,
        )
        eval_results = get_numeric_metrics(eval_results["results"])
        print(f"Obtained evaluation results: {eval_results}")
        return eval_results

    def _load_harness_model(self) -> HFLM | OpenaiCompletionsLM:
        # Instantiate the lm-harness LM class based on the provided model config type
        hf_loader = HuggingFaceAssetLoader(self.artifact_loader)
        match self.config.model:
            case AutoModelConfig() as model_config:
                model_path, revision = hf_loader.resolve_asset_path(model_config.path)
                model_path, peft_path = resolve_peft_and_pretrained(model_path)
                quantization_kwargs: dict[str, Any] = (
                    self.config.quantization.model_dump() if self.config.quantization else {}
                )
                # TODO: Fix this up by passing in the instantiated model directly
                torch_dtype = (
                    self.config.model.torch_dtype if self.config.model.torch_dtype else "auto"
                )
                return HFLM(
                    pretrained=model_path,
                    tokenizer=model_path,
                    peft=peft_path,
                    revision=revision if revision else "main",
                    device="cuda" if torch.cuda.device_count() > 0 else "cpu",
                    trust_remote_code=self.config.model.trust_remote_code,
                    dtype=torch_dtype,
                    **quantization_kwargs,
                )

            case LocalChatCompletionsConfig() as local_config:
                model = local_config.inference.engine
                if isinstance(model, LoadableAssetPath):
                    model, _ = hf_loader.resolve_asset_path(model)
                # If tokenizer is not provided, it is set to the value of model internally
                return OpenaiCompletionsLM(
                    model=model,
                    base_url=local_config.inference.base_url,
                    tokenizer_backend=local_config.tokenizer_backend,
                    truncate=local_config.truncate,
                    max_gen_toks=local_config.max_tokens,
                )

            case _:
                raise ValueError(f"Unexpected model config type: {type(self.config.model)}")
