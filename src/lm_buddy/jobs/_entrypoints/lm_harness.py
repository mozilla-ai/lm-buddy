from typing import Any

import lm_eval
import pandas as pd
import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.openai_completions import OpenaiCompletionsLM

from lm_buddy.integrations.huggingface import AutoModelConfig, HuggingFaceAssetLoader
from lm_buddy.integrations.wandb import ArtifactType, build_table_artifact, default_artifact_name
from lm_buddy.jobs.common import EvaluationResult
from lm_buddy.jobs.configs import LMHarnessJobConfig, LocalChatCompletionsConfig


def get_per_task_dataframes(
    results: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    """Create a `pd.DataFrame` of numeric metrics for each evaluation task.

    This is necessary because artifact tables must have a single datatype for each column.

    lm-harness returns mostly numeric values, but there are also some misc string-valued metrics.
    Filtering down to only numeric values allows us to produce a valid table artifact.
    """
    task_dataframes = {}
    for task_name, data in results.items():
        numeric_rows = [(k, v) for k, v in data.items() if isinstance(v, int | float)]
        task_dataframes[task_name] = pd.DataFrame(data=numeric_rows, columns=["metric", "value"])
    return task_dataframes


def load_harness_model(config: LMHarnessJobConfig) -> HFLM | OpenaiCompletionsLM:
    # Instantiate the lm-harness LM class based on the provided model config type
    hf_loader = HuggingFaceAssetLoader()
    match config.model:
        case AutoModelConfig() as model_config:
            model_path = hf_loader.resolve_asset_path(model_config.path)
            model_path, peft_path = hf_loader.resolve_peft_and_pretrained(model_path)
            quantization_kwargs: dict[str, Any] = (
                config.quantization.model_dump() if config.quantization else {}
            )
            # TODO: Fix this up by passing in the instantiated model directly
            return HFLM(
                pretrained=model_path,
                tokenizer=model_path,
                peft=peft_path,
                device="cuda" if torch.cuda.device_count() > 0 else "cpu",
                trust_remote_code=config.model.trust_remote_code,
                dtype=config.model.torch_dtype if config.model.torch_dtype else "auto",
                **quantization_kwargs,
            )

        case LocalChatCompletionsConfig() as local_config:
            engine_path = hf_loader.resolve_asset_path(local_config.inference.engine)
            return OpenaiCompletionsLM(
                model=engine_path,
                base_url=local_config.inference.base_url,
                tokenizer_backend=local_config.tokenizer_backend,
                truncate=local_config.truncate,
                max_gen_toks=local_config.max_tokens,
            )

        case _:
            raise ValueError(f"Unexpected model config type: {type(config.model)}")


def run_eval(config: LMHarnessJobConfig) -> dict[str, list[tuple[str, float]]]:
    llm = load_harness_model(config)
    eval_results = lm_eval.simple_evaluate(
        model=llm,
        tasks=config.evaluation.tasks,
        batch_size=config.evaluation.batch_size,
        num_fewshot=config.evaluation.num_fewshot,
        limit=config.evaluation.limit,
        log_samples=False,
    )
    print(f"Obtained evaluation results: {eval_results}")
    return get_per_task_dataframes(eval_results["results"])


def run_lm_harness(config: LMHarnessJobConfig) -> EvaluationResult:
    print(f"Running lm-harness evaluation with configuration:\n {config.model_dump_json(indent=2)}")

    base_name = "lm-harness"
    if config.tracking is not None:
        base_name = config.tracking.name
    artifact_name = default_artifact_name(base_name, ArtifactType.EVALUATION)

    eval_tables = run_eval(config)
    table_artifact = build_table_artifact(
        artifact_name=artifact_name,
        artifact_type=ArtifactType.EVALUATION,
        tables=eval_tables,
    )

    return EvaluationResult(
        tables=eval_tables,
        artifacts=[table_artifact],
        dataset_path=None,
    )
