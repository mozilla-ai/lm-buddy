import lm_eval
import ray
import wandb
from lm_eval.models.huggingface import HFLM
from peft import PeftConfig

from flamingo.integrations.wandb import WandbArtifactLoader
from flamingo.integrations.wandb.utils import resolve_artifact_path
from flamingo.jobs import LMHarnessJobConfig


def load_harness_model(config: LMHarnessJobConfig, loader: WandbArtifactLoader) -> HFLM:
    model_path = resolve_artifact_path(config.model.path, loader)

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
        device="cuda" if config.ray.use_gpu else None,
        trust_remote_code=config.model.trust_remote_code,
        dtype=config.model.torch_dtype if config.model.torch_dtype else "auto",
        **quantization_kwargs,
    )


@ray.remote
def evaluation_task(config: LMHarnessJobConfig, model_to_load: str) -> None:
    print("Initializing lm-harness tasks...")
    lm_eval.tasks.initialize_tasks()

    wandb_run = None
    if config.tracking is not None:
        wandb_run = wandb.init(**config.tracking.wandb_init_args(), resume="never")
    artifact_loader = WandbArtifactLoader(wandb_run)

    llm = load_harness_model(config, artifact_loader)
    eval_results = lm_eval.simple_evaluate(
        model=llm,
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        limit=config.limit,
        log_samples=False,
    )
    eval_results = eval_results["results"]
    print(f"Obtained evaluation results: {eval_results}")

    if config.tracking is not None:
        print("Generating table artifact of evaluation results...")
        pass


def run_lm_harness(config: LMHarnessJobConfig):
    print(f"Received job configuration: {config}")

    # Using .options() to dynamically specify resource requirements
    eval_func = evaluation_task.options(num_cpus=config.ray.num_cpus, num_gpus=config.ray.num_gpus)
    eval_future = eval_func.remote(config)

    timeout_seconds = config.timeout.seconds if config.timeout else None
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
