from pydantic import Field

from lm_buddy.integrations.huggingface import TextDatasetConfig
from lm_buddy.integrations.vllm import VLLMCompletionsConfig
from lm_buddy.integrations.wandb import WandbRunConfig
from lm_buddy.jobs.configs import LMBuddyJobConfig
from lm_buddy.types import BaseLMBuddyConfig


class PrometheusEvaluationTaskConfig(BaseLMBuddyConfig):
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


class PrometheusJobConfig(LMBuddyJobConfig):
    """Configuration to run a prometheus job."""

    dataset: TextDatasetConfig = Field(
        description="Dataset of text completions to evaluate using the Prometheus judge model."
    )

    # vLLM endpoint configuration
    prometheus: VLLMCompletionsConfig

    # evaluation task configuration
    evaluation: PrometheusEvaluationTaskConfig | None = None

    # wandb experiment tracking details
    tracking: WandbRunConfig | None = None
