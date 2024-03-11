from typing import Literal

from pydantic import Field, conlist, model_validator

from lm_buddy.types import BaseLMBuddyConfig
from lm_buddy.jobs.configs import LMBuddyJobConfig
from lm_buddy.integrations.wandb import WandbRunConfig
from lm_buddy.integrations.vllm import InferenceServerConfig
from lm_buddy.integrations.huggingface import TextDatasetConfig, AutoTokenizerConfig

class PrometheusCompletionsConfig(BaseLMBuddyConfig):
    """Configuration for a "local-completions" prometheus model.

    The prometheus model is powered by a self-hosted inference server, specified
    as an `InferenceServerConfig`. Additional arguments are also provided
    to control the tokenizer type and generation parameters.
    """

    inference: InferenceServerConfig

    # vLLM-served model params
    best_of: int = 1
    max_tokens: int = 512 
    frequency_penalty: float = 1.03
    temperature: float = 1.0
    top_p: float = 0.9

    # evaluation script params
    tokenizer: AutoTokenizerConfig | None = None
    num_answers: int = 3
    max_retries: int = 5


class PrometheusJobConfig(LMBuddyJobConfig):
    """Configuration to run a prometheus evaluation job."""

    dataset: TextDatasetConfig = Field(description="dataset (json artifact from which we'll extract `text_field`)")
    # details for our self-hosted prometheus endpoint
    prometheus: PrometheusCompletionsConfig
    # wandb experiment tracking details
    tracking: WandbRunConfig | None = None
