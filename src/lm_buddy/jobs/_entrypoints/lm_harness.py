from typing import Any

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.openai_completions import OpenaiCompletionsLM

from lm_buddy.integrations.huggingface import (
    AutoModelConfig,
    HuggingFaceAssetLoader,
    HuggingFaceAssetPath,
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
from lm_buddy.jobs.common import LMBuddyJobType
from lm_buddy.jobs.configs import LMHarnessJobConfig, LocalChatCompletionsConfig


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


def load_harness_model(
    config: LMHarnessJobConfig,
    artifact_loader: ArtifactLoader,
) -> HFLM | OpenaiCompletionsLM:
    # Instantiate the lm-harness LM class based on the provided model config type
    hf_loader = HuggingFaceAssetLoader(artifact_loader)
    match config.model:
        case AutoModelConfig() as model_config:
            model_path, revision = hf_loader.resolve_asset_path(model_config.load_from)
            model_path, peft_path = resolve_peft_and_pretrained(model_path)
            quantization_kwargs: dict[str, Any] = (
                config.quantization.model_dump() if config.quantization else {}
            )
            # TODO: Fix this up by passing in the instantiated model directly
            return HFLM(
                pretrained=model_path,
                tokenizer=model_path,
                peft=peft_path,
                revision=revision if revision else "main",
                device="cuda" if torch.cuda.device_count() > 0 else "cpu",
                trust_remote_code=config.model.trust_remote_code,
                dtype=config.model.torch_dtype if config.model.torch_dtype else "auto",
                **quantization_kwargs,
            )

        case LocalChatCompletionsConfig() as local_config:
            model = local_config.inference.engine
            if isinstance(model, HuggingFaceAssetPath):
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
            raise ValueError(f"Unexpected model config type: {type(config.model)}")


def load_and_evaluate(
    config: LMHarnessJobConfig,
    artifact_loader: ArtifactLoader,
) -> dict[str, list[tuple[str, float]]]:
    print("Initializing lm-harness tasks...")

    llm = load_harness_model(config, artifact_loader)
    eval_results = lm_eval.simple_evaluate(
        model=llm,
        tasks=config.evaluator.tasks,
        batch_size=config.evaluator.batch_size,
        num_fewshot=config.evaluator.num_fewshot,
        limit=config.evaluator.limit,
        log_samples=False,
    )
    eval_results = get_numeric_metrics(eval_results["results"])
    print(f"Obtained evaluation results: {eval_results}")
    return eval_results


def run_lm_harness(config: LMHarnessJobConfig, artifact_loader: ArtifactLoader):
    print(f"Received job configuration:\n {config.model_dump_json(indent=2)}")

    if config.tracking is not None:
        with wandb_init_from_config(
            config.tracking,
            parameters=config.evaluator,  # Log eval settings in W&B run
            resume=WandbResumeMode.ALLOW,
            job_type=LMBuddyJobType.EVALUATION,
        ) as run:
            eval_results = load_and_evaluate(config, artifact_loader)
            eval_artifact = build_table_artifact(
                artifact_name=default_artifact_name(run.name, ArtifactType.EVALUATION),
                artifact_type=ArtifactType.EVALUATION,
                columns=["metric", "value"],
                tables=eval_results,
            )
            print("Logging artifact for evaluation results...")
            artifact_loader.log_artifact(eval_artifact)
    else:
        load_and_evaluate(config, artifact_loader)
