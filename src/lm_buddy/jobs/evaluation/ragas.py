from pathlib import Path

from datasets import load_dataset
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from lm_buddy.configs.jobs.ragas import RagasJobConfig
from lm_buddy.jobs.asset_loader import HuggingFaceAssetLoader
from lm_buddy.jobs.common import EvaluationResult
from lm_buddy.preprocessing import format_dataset_with_prompt
from lm_buddy.tracking.artifact_utils import (
    ArtifactType,
    build_directory_artifact,
    default_artifact_name,
)

RAGAS_METRICS_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_recall": context_recall,
    "context_precision": context_precision,
}


def run_eval(config: RagasJobConfig) -> Path:
    # load dataset from W&B artifact
    hf_loader = HuggingFaceAssetLoader()
    evaluation_dataset = hf_loader.load_dataset(config.dataset)
    if config.dataset.prompt_template is not None:
        evaluation_dataset = format_dataset_with_prompt(
            evaluation_dataset, config.dataset.prompt_template, config.dataset.text_field
        )

    # ragas custom model args
    ragas_args = {}

    # load embedding model
    embedding_model = hf_loader.resolve_asset_path(config.evaluation.embedding_model.path)
    ragas_args["embeddings"] = HuggingFaceEmbeddings(model_name=embedding_model)

    # configure ragas to point to vllm instance for generation
    inference_engine = hf_loader.resolve_asset_path(config.judge.inference.engine)
    ragas_args["llm"] = ChatOpenAI(
        model=inference_engine,
        openai_api_key="EMPTY",  # needed to hit custom openai endpoint
        openai_api_base=config.judge.inference.base_url,
        max_tokens=config.judge.max_tokens,
        temperature=config.judge.temperature,
        top_k=config.judge.top_k,
    )

    result = ragas_evaluate(
        dataset=evaluation_dataset,
        metrics=RAGAS_METRICS_MAP[config.evaluation.metrics],
        **ragas_args,
    )
    result_df = result.to_pandas()

    # open the output file for writing and iterate on samples
    tracking_name = config.tracking.name if config.tracking is not None else "output.json"
    output_fname = Path(config.evaluation.output_folder) / tracking_name
    result_df.to_json(output_fname)

    # convert plain json dataset in HF format
    output_dataset_path = Path(config.evaluation.output_folder) / "hf" / tracking_name
    ds = load_dataset("json", data_files=str(output_fname), split="train")
    ds.save_to_disk(output_dataset_path)

    return output_dataset_path


def run_ragas(config: RagasJobConfig) -> EvaluationResult:
    output_dataset_path = run_eval(config)
    print(f"Ragas evaluation dataset stored at {output_dataset_path}")

    # Create a directory artifact for the HF dataset
    artifact_name = default_artifact_name(config.name, artifact_type=ArtifactType.DATASET)
    dataset_artifact = build_directory_artifact(
        artifact_name=artifact_name,
        artifact_type=ArtifactType.DATASET,
        dir_path=output_dataset_path,
        reference=False,
    )

    return EvaluationResult(
        artifacts=[dataset_artifact],
        dataset_path=output_dataset_path,
        tables={},
    )
