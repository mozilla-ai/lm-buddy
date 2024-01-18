from typing import Any

import lm_eval
import ray
import wandb
from lm_eval.models.huggingface import HFLM
from peft import PeftConfig

from flamingo.integrations.wandb import ArtifactType, WandbArtifactLoader
from flamingo.integrations.wandb.utils import default_artifact_name, wandb_init_from_config
from flamingo.jobs.lm_harness import LMHarnessJobConfig


def build_evaluation_artifact(run_name: str, results: dict[str, dict[str, Any]]) -> wandb.Artifact:
    print("Building artifact for evaluation results...")
    artifact_name = default_artifact_name(run_name, ArtifactType.EVALUATION)
    artifact = wandb.Artifact(artifact_name, type=ArtifactType.EVALUATION.value)
    for task_name, task_results in results.items():
        # Filter down to numeric metrics from task dict
        task_data = [(k, v) for k, v in task_results.items() if isinstance(v, int | float)]
        task_table = wandb.Table(data=task_data, columns=["metric", "value"])
        artifact.add(task_table, name=f"task-{task_name}")
    return artifact


def load_harness_model(config: LMHarnessJobConfig, loader: WandbArtifactLoader) -> HFLM:
    model_path = loader.resolve_path_reference(config.model.path)

    # We don't know if the checkpoint is adapter weights or merged model weights
    # Try to load as an adapter and fall back to the checkpoint containing the full model
    try:
        adapter_config = PeftConfig.from_pretrained(model_path)
        pretrained = adapter_config.base_model_name_or_path
        peft = model_path
    except ValueError as e:
        print(
            f"Unable to load model as adapter: {e}. "
            "This is expected if the checkpoint does not contain adapter weights."
        )
        pretrained = model_path
        peft = None

    # Return lm-harness model wrapper class
    quantization_kwargs = config.quantization.dict() if config.quantization else {}
    return HFLM(
        pretrained=pretrained,
        tokenizer=pretrained,
        peft=peft,
        device="cuda" if config.ray.num_gpus > 0 else None,
        trust_remote_code=config.model.trust_remote_code,
        dtype=config.model.torch_dtype if config.model.torch_dtype else "auto",
        **quantization_kwargs,
    )


def evaluate_with_loader(config: LMHarnessJobConfig, loader: WandbArtifactLoader) -> dict[str, Any]:
    print("Initializing lm-harness tasks...")
    lm_eval.tasks.initialize_tasks()

    llm = load_harness_model(config, loader)
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
        with wandb_init_from_config(config.tracking, resume="never") as run:
            artifact_loader = WandbArtifactLoader(run=run)
            eval_results = evaluate_with_loader(config, artifact_loader)
            artifact = build_evaluation_artifact(run.name, eval_results)
            run.log_artifact(artifact)
    else:
        artifact_loader = WandbArtifactLoader(run=None)
        evaluate_with_loader(config, artifact_loader)


def run_lm_harness(config: LMHarnessJobConfig):
    print(f"Received job configuration:\n {config.json(indent=2)}")

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
