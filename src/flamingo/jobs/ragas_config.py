import datetime
from pathlib import Path

from pydantic import validator
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)
from ragas.metrics.base import MetricWithLLM

from flamingo.jobs import BaseJobConfig


class RagasEvaluationConfig(BaseJobConfig):
    """Configuration to run a Ragas evaluation job.

    This job loads a dataset from an existing path on our cluster.
    The dataset must be formatted in the RAG context, with the question, generated answer,
    the contexts (retrieved), and optionally a ground truth field.
    """

    is_hf_dataset: bool | None = False
    data_path: str | Path | None = None

    # remap columns so data table does not need to be edited
    data_column_names: dict = {
        "question": "question",
        "answer": "answer",
        "contexts": "contexts",
    }

    metrics: list[MetricWithLLM] = [
        faithfulness,
        answer_relevancy,
        # context_recall,
        context_precision,
    ]

    result_path: str | Path | None = None

    limit: int | float | None = None
    num_cpus: int = 1
    num_gpus: int = 1
    timeout: datetime.timedelta | None = None

    @validator("data_column_names")
    def _validate_data_column_names(cls, v):
        required_keys = {"question", "answer", "contexts"}
        provided_keys = set(v.keys())
        missing_keys = required_keys.difference(provided_keys)
        if not missing_keys:
            return v
        else:
            raise ValueError(f"`{v}` is missing the following keys:[{missing_keys}]")
