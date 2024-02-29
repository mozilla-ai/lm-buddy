import os
from pathlib import Path

import ray
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from langchain.chat_models import ChatOpenAI
from langchain_core.embeddings import Embeddings
from ragas import evaluate

from lm_buddy.integrations.wandb import get_wandb_summary, update_wandb_summary
from lm_buddy.jobs.ragas import RagasEvaluationJobConfig


def resolve_data_path(config: RagasEvaluationJobConfig) -> str:
    data_path = None
    if config.data_path:
        print("Attempting to resolve data path from existing W&B data processing run...")
        run_summary = get_wandb_summary(config.wandb_env)
        path = Path(run_summary["dataset_path"])
        print(f"Using data path from wandb run: {path}")
        if not path.exists():
            raise (FileNotFoundError(f"{path} cannot be found."))
        data_path = str(path)
    else:
        data_path = str(config.data_path)
    print(f"Dataset directory path: {data_path}")
    return data_path


def _load_dataset_for_ragas_eval(
    config: RagasEvaluationJobConfig,
) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    evaluation_dataset_to_load = config.dataset.data_path

    print(f"Loading dataset from {evaluation_dataset_to_load}")
    dataset = (
        load_dataset(evaluation_dataset_to_load)
        if config.is_hf_dataset
        else load_dataset("parquet", data_files=evaluation_dataset_to_load)
    )

    print(
        f"Remapping data columns to be ragas compatible with \
          the following {config.data_column_names}"
    )
    dataset.rename_column("question", config.data_column_names["question"])
    dataset.rename_column("answer", config.data_column_names["answer"])
    dataset.rename_column("contexts", config.data_column_names["contexts"])

    return dataset


def evaluation_task(config: RagasEvaluationJobConfig) -> None:
    # ragas custom model args
    ragas_args = {}

    # set up custom embedding model for ragas (only supports langchain embedding models right now)
    if config.judge_model.embedding_model:
        ragas_args["embeddings"] = Embeddings(
            model=config.judge_model.embedding_model
        )  # any langchain embedding instance

    # set up custom judge LLM model (called from vllm server)
    if config.judge_model.language_model:
        # create vLLM Langchain instance
        vllm_entry = ChatOpenAI(
            model=config.judge_model.language_model,
            openai_api_key=config.judge_model.openai_api_key,
            # get api endpoint from environment variable
            openai_api_base=os.environ.get("VLLM_JUDGE_ENDPOINT"),
            max_tokens=config.judge_model.max_tokens,
            temperature=config.judge_model.temperature,
        )

        ragas_args["llm"] = vllm_entry

    dataset = _load_dataset_for_ragas_eval(config)

    print("Initializing ragas eval task...")
    result = evaluate(dataset=dataset, metrics=config.metrics, **ragas_args)

    print(f"Obtained evaluation results: {result}")

    if config.wandb_env:
        print("Logging results to W&B...")
        update_wandb_summary(config.wandb_env, result)


def run_ragas_evaluation(config: RagasEvaluationJobConfig):
    print(f"Received job configuration: {config}")

    # Resolve path and ensure exists
    evaluation_dataset_to_load = resolve_data_path(config)

    # Using .options() to dynamically specify resource requirements
    eval_func = evaluation_task.options(num_gpus=config.num_gpus)
    eval_future = eval_func.remote(config, evaluation_dataset_to_load)

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
