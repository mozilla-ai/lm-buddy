from pathlib import Path

import lm_eval
import ray
from lm_eval.models.huggingface import HFLM
from peft import PeftConfig
from tuner.integrations.wandb import get_wandb_summary, update_wandb_summary

from flamingo.jobs import LMHarnessJobConfig, ModelNameOrCheckpointPath


def resolve_model_or_path(config: LMHarnessJobConfig) -> str:
    mn_or_path = None
    match config.model_name_or_path:
        case None:
            print("Attempting to resolve checkpoint path from existing W&B run...")
            run_summary = get_wandb_summary(config.wandb_env)
            cp = Path(run_summary["ray/checkpoint_path"])
            print(f"Using checkpoint path from wandb run: {cp}")
            if not cp.exists():
                raise (FileNotFoundError(f"{mn_or_path} cannot be found."))
            mn_or_path = str(cp)
        case ModelNameOrCheckpointPath(checkpoint=None) as x:
            print("No checkpoint; will attempt to load model from HuggingFace")
            mn_or_path = x.name
        case ModelNameOrCheckpointPath(checkpoint=ckpt):
            print(f"Checkpoint found; will attempt to load model from {ckpt}")
            mn_or_path = ckpt
        case _:
            raise (
                ValueError(
                    "Something is wrong with the passed "
                    f"model_name_or_path: {config.model_name_or_path}"
                )
            )
    return mn_or_path


def load_harness_model(config: LMHarnessJobConfig, model_to_load: str) -> HFLM:
    # We don't know if the checkpoint is adapter weights or merged model weights
    # Try to load as an adapter and fall back to the checkpoint containing the full model
    try:
        adapter_config = PeftConfig.from_pretrained(model_to_load)
        pretrained = adapter_config.base_model_name_or_path
        peft = model_to_load
    except ValueError as e:
        print(
            f"Unable to load model as adapter: {e}. "
            "This is expected if the checkpoint does not contain adapter weights."
        )
        pretrained = model_to_load
        peft = None

    # Return lm-harness model wrapper class
    quantization_kwargs = config.quantization_config.dict() if config.quantization_config else {}
    return HFLM(
        pretrained=pretrained,
        tokenizer=pretrained,
        peft=peft,
        device="cuda" if config.num_gpus > 0 else None,
        trust_remote_code=config.trust_remote_code,
        dtype=config.torch_dtype if config.torch_dtype else "auto",
        **quantization_kwargs,
    )


@ray.remote
def run_evaluation(config: LMHarnessJobConfig, model_to_load: str) -> None:
    print("Initializing lm-harness tasks...")
    lm_eval.tasks.initialize_tasks()

    print("Running lm-harness evaluation inside remote function...")
    llm = load_harness_model(config, model_to_load)
    raw_results = lm_eval.simple_evaluate(
        model=llm,
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        limit=config.limit,
        log_samples=False,
    )
    print("Finished lm-harness evaluation inside remote function")

    formatted_results = {}
    for task_name, metrics in raw_results["results"].items():
        task_metrics = {
            f"{task_name}/{metric.replace(',', '_')}": value for metric, value in metrics.items()
        }
        formatted_results.update(task_metrics)
    print(f"Obtained evaluation results: {formatted_results}")

    if config.wandb_env:
        print("Logging results to W&B...")
        update_wandb_summary(config.wandb_env, formatted_results)


def main(config: LMHarnessJobConfig):
    print(f"Received job configuration: {config}")

    # Resolve path and ensure exists
    model_to_load = resolve_model_or_path(config)

    # Using .options() to dynamically specify resource requirements
    eval_func = run_evaluation.options(num_cpus=config.num_cpus, num_gpus=config.num_gpus)
    eval_future = eval_func.remote(config, model_to_load)

    timeout_seconds = config.timeout.seconds if config.timeout else None
    try:
        print("Waiting on evaluation task...")
        ray.get(eval_future, timeout=timeout_seconds)
        print("Evaluation successfully completed")
    except TimeoutError:
        print(
            f"Evaluation task timed out after {timeout_seconds} sec. "
            "If the evaluation runner finished but the task failed to shut down, "
            "please check if your results were still generated and persisted."
        )
        raise
