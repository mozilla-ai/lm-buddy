from typing import Literal

from pydantic import Field, field_validator

from lm_buddy.integrations.huggingface import AutoModelConfig
from lm_buddy.integrations.huggingface.dataset_config import TextDatasetConfig
from lm_buddy.integrations.vllm import VLLMCompletionsConfig
from lm_buddy.paths import AssetPath
from lm_buddy.types import BaseLMBuddyConfig

RagasEvaluationMetric = Literal[
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
]


class RagasEvaluationConfig(BaseLMBuddyConfig):
    """Parameters specifically required for RAGAs Evaluation"""

    metrics: list[RagasEvaluationMetric] = Field(
        default=[
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
        ]
    )

    # language model and embedding models used as evaluation judges
    embedding_model: AutoModelConfig | None = "sentence-transformers/all-mpnet-base-v2"

    # path to store the generated ratings/evaluations of each dataset sample
    output_folder: str = "/tmp"

    @field_validator("embedding_model", mode="before")
    def validate_embedding_model_arg(cls, x):
        """Allow for passing just a path string as the model argument."""
        if isinstance(x, str):
            return AutoModelConfig(path=x)
        return x


class RagasJobConfig(BaseLMBuddyConfig):
    """Configuration to run a Ragas evaluation job.

    This job loads a dataset from an existing path on our cluster.
    The dataset must be formatted in the RAG context, with the question, generated answer,
    the contexts (retrieved), and optionally a ground truth field.
    """

    judge: VLLMCompletionsConfig = Field(description="Externally hosted Ragas judge model.")
    dataset: TextDatasetConfig = Field(
        description="Dataset of text completions to evaluate using the Ragas judge model."
    )
    evaluation: RagasEvaluationConfig = Field(
        default_factory=RagasEvaluationConfig,
        description="Settings for the Ragas evaluation.",
    )

    def asset_paths(self) -> set[AssetPath]:
        paths = {self.dataset.path}
        if self.judge.inference.engine is not None:
            paths.add(self.judge.inference.engine)
        return paths
