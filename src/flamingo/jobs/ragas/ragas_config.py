import datetime
from pathlib import Path

from pydantic import Field, validator
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.metrics.base import MetricWithLLM

from flamingo.integrations.wandb import WandbRunConfig
from flamingo.types import BaseFlamingoConfig


class RagasEvaluationDatasetConfig(BaseFlamingoConfig):
    """to configure the evaluation dataset source"""

    is_hf_dataset: bool | None = False
    data_path: str | Path | None = None

    # remap columns so data table does not need to be edited
    data_column_names: dict = {
        "question": "question",
        "answer": "answer",
        "contexts": "contexts",
    }

    @validator("data_column_names")
    def _validate_data_column_names(cls, v):
        required_keys = {"question", "answer", "contexts"}
        provided_keys = set(v.keys())
        missing_keys = required_keys.difference(provided_keys)
        if not missing_keys:
            return v
        else:
            raise ValueError(f"`{v}` is missing the following keys:[{missing_keys}]")


class RagasRayConfig(BaseFlamingoConfig):
    """Misc settings for Ray compute for ragas eval job."""

    num_cpus: int | float = 1
    num_gpus: int | float = 1
    timeout: datetime.timedelta | None = None


class RagasEvalConfig(BaseFlamingoConfig):
    "misc settings for ragas eval job"

    metrics: list[MetricWithLLM] = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    # path to store the generated ratings/evaluations of each dataset sample
    result_path: str | Path | None = None


class RagasEvaluatioJobConfig(BaseFlamingoConfig):
    """Configuration to run a Ragas evaluation job.

    This job loads a dataset from an existing path on our cluster.
    The dataset must be formatted in the RAG context, with the question, generated answer,
    the contexts (retrieved), and optionally a ground truth field.
    """

    # evaluation settings for ragas
    dataset: RagasEvaluationDatasetConfig
    evaluator: RagasEvalConfig

    # wandb model run to associate to the ragas evaluator
    tracking: WandbRunConfig | None = None

    ray: RagasRayConfig = Field(default_factory=RagasRayConfig)
