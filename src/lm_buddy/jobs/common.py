from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from datasets import Dataset

from lm_buddy.integrations.huggingface import TextDatasetConfig
from lm_buddy.preprocessing import format_dataset_with_prompt


class LMBuddyJobType(str, Enum):
    """Enumeration of logical job types runnable via the LM Buddy."""

    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class JobResult:
    artifacts: list[wandb.Artifact]


@dataclass
class FinetuningResult(JobResult):
    """Result from a finetuning task."""

    checkpoint_path: Path | None
    metrics: dict[str, Any] | None
    is_adapter: bool


@dataclass
class EvaluationResult(JobResult):
    """Result from an evaluation task, containing aggregate metrics and artifacts."""

    tables: dict[str, pd.DataFrame]
    dataset_path: Path | None


def preprocess_text_dataset(dataset: Dataset, dataset_config: TextDatasetConfig) -> Dataset:
    """Prompt format a text dataset if a prompt template is specified on the config."""
    if dataset_config.prompt_template is not None:
        return format_dataset_with_prompt(
            dataset=dataset,
            template=dataset_config.prompt_template,
            output_field=dataset_config.text_field,
        )
    else:
        return dataset
