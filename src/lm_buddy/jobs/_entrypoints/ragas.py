from pathlib import Path

from datasets import load_dataset
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from lm_buddy.integrations.huggingface import HuggingFaceAssetLoader
from lm_buddy.integrations.wandb import (
    ArtifactLoader,
    ArtifactType,
    build_directory_artifact,
    default_artifact_name,
)
from lm_buddy.integrations.wandb.run_utils import wandb_init_from_config
from lm_buddy.jobs._entrypoints.utils import preprocess_text_dataset
from lm_buddy.jobs.common import EvaluationResult, LMBuddyJobType
from lm_buddy.jobs.configs import RagasJobConfig
from lm_buddy.paths import AssetPath

RAGAS_METRICS_MAP = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_recall": context_recall,
    "context_precision": context_precision,
}


def run_eval(config: RagasJobConfig, artifact_loader: ArtifactLoader) -> AssetPath:
    # load dataset from W&B artifact
    hf_loader = HuggingFaceAssetLoader(artifact_loader)
    evaluation_dataset = hf_loader.load_dataset(config.dataset)
    evaluation_dataset = preprocess_text_dataset(evaluation_dataset, config.dataset)

    # ragas custom model args
    ragas_args = {}

    # load embedding model
    embedding_model = hf_loader.resolve_asset_path(config.evaluation.embedding_model.path)
    ragas_args["embeddings"] = HuggingFaceEmbeddings(model_name=embedding_model.strip_prefix())

    # configure ragas to point to vllm instance for generation
    inference_engine = hf_loader.resolve_asset_path(config.judge.inference.engine)
    ragas_args["llm"] = ChatOpenAI(
        model=inference_engine.strip_prefix(),
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

    return AssetPath.from_file(output_dataset_path)


def run_ragas(config: RagasJobConfig, artifact_loader: ArtifactLoader) -> EvaluationResult:
    # Run ragas eval and store output in local filename
    if config.tracking:
        with wandb_init_from_config(config.tracking, job_type=LMBuddyJobType.EVALUATION) as run:
            output_file_path = run_eval(config, artifact_loader)

            # Create a directory artifact for the HF dataset
            dataset_artifact = build_directory_artifact(
                dir_path=output_file_path,
                artifact_name=default_artifact_name(run.name, artifact_type=ArtifactType.DATASET),
                artifact_type=ArtifactType.DATASET,
                reference=False,
            )
            dataset_artifact_path = AssetPath.from_wandb(
                dataset_artifact.name, run.project, run.entity
            )

            print("Logging dataset artifact for Ragas evaluation ...")
            artifact_loader.log_artifact(dataset_artifact)

    else:
        output_file_path = run_eval(config, artifact_loader)
        dataset_artifact_path = None

    print(f"Evaluation dataset stored at {output_file_path}")
    return EvaluationResult(
        tables={},
        table_artifact_path=None,
        dataset_path=output_file_path,
        dataset_artifact_path=dataset_artifact_path,
    )
