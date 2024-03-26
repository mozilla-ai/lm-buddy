from typing import Literal

from pydantic import Field, field_validator

from lm_buddy.integrations.huggingface import AutoModelConfig
from lm_buddy.integrations.huggingface.dataset_config import TextDatasetConfig
from lm_buddy.integrations.vllm import VLLMCompletionsConfig
from lm_buddy.integrations.wandb import WandbRunConfig
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
        default_factory=[
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
        ]
    )

    # openAI API key if using openAI for judge models
    openai_api_key: str | None = "EMPTY"

    # language model and embedding models used as evaluation judges
    embedding_model: AutoModelConfig | None = "sentence-transformers/all-mpnet-base-v2"

    # path to store the generated ratings/evaluations of each dataset sample
    output_folder: str = "/tmp"

    @field_validator("embedding_model", mode="before", always=True)
    def validate_embedding_model_arg(cls, x):
        """Allow for passing just a path string as the model argument."""
        if isinstance(x, str):
            return AutoModelConfig(load_from=x)
        return x


class RagasJobConfig(BaseLMBuddyConfig):
    """Configuration to run a Ragas evaluation job.

    This job loads a dataset from an existing path on our cluster.
    The dataset must be formatted in the RAG context, with the question, generated answer,
    the contexts (retrieved), and optionally a ground truth field.
    """

    # vllm inference server for generation
    judge: VLLMCompletionsConfig = Field(description="Externally hosted Ragas judge model.")

    # dataset containing the relevant fields required for ragas evaluation
    dataset: TextDatasetConfig = Field(
        description="Dataset of text completions to evaluate using the Ragas judge model."
    )

    # evaluation settings for ragas
    evaluation: RagasEvaluationConfig = Field(
        default_factory=RagasEvaluationConfig,
        description="Settings for the Ragas evaluation.",
    )

    # wandb model run to associate to the ragas evaluator
    tracking: WandbRunConfig | None = None
