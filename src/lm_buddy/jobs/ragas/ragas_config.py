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

from flamingo.integrations.huggingface import AutoModelConfig
from flamingo.integrations.wandb import WandbRunConfig
from flamingo.types import BaseFlamingoConfig


class RagasConfig(BaseFlamingoConfig):
    """misc settings for ragas eval job"""

    metrics: list[MetricWithLLM] = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ]

    # path to store the generated ratings/evaluations of each dataset sample
    result_path: str | Path | None = None


class RagasvLLMJudgeConfig(BaseFlamingoConfig):
    """
    Configuration class for a vLLM hosted judge model
    Requires a vLLM endpoint that the model will hit instead of the openAI default
    """

    model: AutoModelConfig
    inference_server_url: str | None = "http://localhost:8080/v1"
    openai_api_key: str | None = "no-key"
    max_tokens: int | None = 5
    temperature: float | None = 0

    @validator("model", pre=True, always=True)
    def validate_model_arg(cls, x):
        """Allow for passing just a path string as the model argument."""
        if isinstance(x, str):
            return AutoModelConfig(load_from=x)
        return x


class RagasRayConfig(BaseFlamingoConfig):
    """Misc settings for Ray compute for ragas eval job."""

    num_cpus: int | float = 1
    num_gpus: int | float = 1
    timeout: datetime.timedelta | None = None


class RagasEvaluationDatasetConfig(BaseFlamingoConfig):
    """to configure the evaluation dataset source"""

    data_path: str | Path | None = None

    # columns of relevant maps
    question_col: str | None = "question"
    answer_col: str | None = "answer"
    context_col: str | None = "contexts"
    ground_truth_col: str | None = None


class RagasEvaluationJobConfig(BaseFlamingoConfig):
    """Configuration to run a Ragas evaluation job.

    This job loads a dataset from an existing path on our cluster.
    The dataset must be formatted in the RAG context, with the question, generated answer,
    the contexts (retrieved), and optionally a ground truth field.
    """

    # evaluation settings for ragas
    dataset: RagasEvaluationDatasetConfig
    evaluator: RagasConfig

    # Judge LLM settings
    judge_model: RagasvLLMJudgeConfig | None = None

    # wandb model run to associate to the ragas evaluator
    tracking: WandbRunConfig | None = None

    # ray settings
    ray: RagasRayConfig = Field(default_factory=RagasRayConfig)
