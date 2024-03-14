from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

from lm_buddy.integrations.wandb import WandbArtifactConfig


class LMBuddyJobType(str, Enum):
    """Enumeration of logical job types runnable via the LM Buddy."""

    SIMPLE = "simple"
    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class SimpleOutput:
    """Output from a simple test task."""

    magic_number: int


@dataclass
class FinetuningOutput:
    """Output from a finetuning task."""

    artifact: WandbArtifactConfig | None
    checkpoint_path: Path | None
    metrics: dict[str, float]
    is_adapter: bool


@dataclass
class EvaluationOutput:
    """Output from an evaluation task containing aggregate metrics and artifact locations."""

    results: dict[str, pd.DataFrame]
    results_artifact: WandbArtifactConfig | None
    dataset_artifact: WandbArtifactConfig | None
    dataset_path: Path | None
