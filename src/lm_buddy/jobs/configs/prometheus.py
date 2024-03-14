from pydantic import Field

from lm_buddy.integrations.huggingface import TextDatasetConfig
from lm_buddy.integrations.vllm import VLLMCompletionsConfig
from lm_buddy.integrations.wandb import WandbRunConfig
from lm_buddy.jobs.configs import LMBuddyJobConfig
from lm_buddy.types import BaseLMBuddyConfig


class PrometheusEvaluationConfig(BaseLMBuddyConfig):
    """Parameters specific to Prometheus evaluation."""

    num_answers: int = 3
    max_retries: int = 5
    scores: list = [1, 2, 3, 4, 5]
    min_score: int = 0
    max_score: int = 5
    enable_tqdm: bool = False
    output_folder: str = "/tmp"
    conversation_template: str = "llama-2"
    conversation_system_message: str = "You are a fair evaluator language model."


class PrometheusTaskConfig(LMBuddyJobConfig):
    """Configuration for a Prometheus judge evaluation task."""

    prometheus: VLLMCompletionsConfig = Field(
        description="Externally hosted Prometheus judge model."
    )
    dataset: TextDatasetConfig = Field(
        description="Dataset of text completions to evaluate using the Prometheus judge model."
    )
    evaluation: PrometheusEvaluationConfig = Field(
        default_factory=PrometheusEvaluationConfig,
        description="Settings for the Prometheus evaluation.",
    )
    tracking: WandbRunConfig | None = None
