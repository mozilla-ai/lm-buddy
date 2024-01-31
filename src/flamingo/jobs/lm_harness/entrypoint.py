from typing import Any

import lm_eval
import ray
import wandb
from lm_eval.models.huggingface import HFLM
from lm_eval.models.openai_completions import OpenaiCompletionsLM
from peft import PeftConfig

from flamingo.integrations.huggingface import AutoModelConfig, resolve_loadable_path
from flamingo.integrations.vllm import InferenceServerConfig
from flamingo.integrations.wandb import (
    ArtifactType,
    WandbResumeMode,
    default_artifact_name,
    wandb_init_from_config,
)
from flamingo.jobs.lm_harness import LMHarnessJobConfig
from flamingo.jobs.utils import FlamingoJobType


# TODO: Should this also be abstracted to a helper method like log_artifact_from_path?
def log_evaluation_artifact(run_name: str, results: dict[str, dict[str, Any]]) -> wandb.Artifact:
    print("Building artifact for evaluation results...")
    artifact_name = default_artifact_name(run_name, ArtifactType.EVALUATION)
    artifact = wandb.Artifact(artifact_name, type=ArtifactType.EVALUATION)
    for task_name, task_results in results.items():
        # Filter down to numeric metrics from task dict
        task_data = [(k, v) for k, v in task_results.items() if isinstance(v, int | float)]
        task_table = wandb.Table(data=task_data, columns=["metric", "value"])
        artifact.add(task_table, name=f"task-{task_name}")
    return wandb.log_artifact(artifact)


def load_harness_model(config: LMHarnessJobConfig) -> HFLM | OpenaiCompletionsLM:
    if isinstance(config.model, AutoModelConfig):
        # We don't know if the checkpoint is adapter weights or merged model weights
        # Try to load as an adapter and fall back to the checkpoint containing the full model
        path, revision = resolve_loadable_path(config.model.load_from)
        try:
            peft_config = PeftConfig.from_pretrained(path, revision=revision)
            peft_path = path
            pretrained_model_path = peft_config.base_model_name_or_path
        except ValueError as e:
            print(
                f"Unable to load model as adapter: {e}. "
                "This is expected if the checkpoint does not contain adapter weights."
            )
            peft_path = None
            pretrained_model_path = path

        # Return the lm-harness version of a HuggingFace LLM
        quantization_kwargs = config.quantization.dict() if config.quantization else {}
        return HFLM(
            pretrained=pretrained_model_path,
            tokenizer=pretrained_model_path,
            peft=peft_path,
            revision=revision,
            device="cuda" if config.ray.num_gpus > 0 else None,
            trust_remote_code=config.model.trust_remote_code,
            dtype=config.model.torch_dtype if config.model.torch_dtype else "auto",
            **quantization_kwargs,
        )

    elif isinstance(config.model, InferenceServerConfig):
        # Return the lm-harness version of a model endpoint
        return OpenaiCompletionsLM(
            model=config.model.model_name,
            tokenizer=config.model.tokenizer,
            base_url=config.model.base_url,
            tokenizer_backend=config.model.tokenizer_backend,


    else:
        raise ValueError(f"Unexpected model config type: {type(config.model)}")


def load_and_evaluate(config: LMHarnessJobConfig) -> dict[str, Any]:
    print("Initializing lm-harness tasks...")
    lm_eval.tasks.initialize_tasks()

    llm = load_harness_model(config)
    eval_results = lm_eval.simple_evaluate(
        model=llm,
        tasks=config.evaluator.tasks,
        batch_size=config.evaluator.batch_size,
        num_fewshot=config.evaluator.num_fewshot,
        limit=config.evaluator.limit,
        log_samples=False,
    )
    eval_results = eval_results["results"]
    print(f"Obtained evaluation results: {eval_results}")
    return eval_results


@ray.remote
def evaluation_task(config: LMHarnessJobConfig) -> None:
    if config.tracking is not None:
        with wandb_init_from_config(
            config.tracking,
            parameters=config.evaluator,  # Log eval settings in W&B run
            resume=WandbResumeMode.ALLOW,
            job_type=FlamingoJobType.EVALUATION,
        ) as run:
            eval_results = load_and_evaluate(config)
            log_evaluation_artifact(run.name, eval_results)
    else:
        load_and_evaluate(config)


def run_lm_harness(config: LMHarnessJobConfig):
    print(f"Received job configuration:\n {config.model_dump_json(indent=2)}")

    # Using .options() to dynamically specify resource requirements
    eval_func = evaluation_task.options(num_cpus=config.ray.num_cpus, num_gpus=config.ray.num_gpus)
    eval_future = eval_func.remote(config)

    timeout_seconds = config.ray.timeout.seconds if config.ray.timeout else None
    try:
        print("Waiting on evaluation task...")
        ray.get(eval_future, timeout=timeout_seconds)
        print("Evaluation successfully completed!")
    except TimeoutError:
        print(
            f"Evaluation task timed out after {timeout_seconds} sec. "
            "If the evaluation runner finished but the task failed to shut down, "
            "please check if your results were still generated and persisted."
        )
        raise
