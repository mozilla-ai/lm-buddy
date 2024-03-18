from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from lm_buddy.integrations.wandb import WandbArtifactConfig


class LMBuddyJobType(str, Enum):
    """Enumeration of logical job types runnable via the LM Buddy."""

    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class FinetuningResult:
    """Result from a finetuning task."""

    checkpoint_path: Path | None
    checkpoint_artifact: WandbArtifactConfig | None
    metrics: dict[str, Any]
    is_adapter: bool


@dataclass
class EvaluationResult:
    """Result from an evaluation task, containing aggregate metrics and artifacts."""

    tables: dict[str, pd.DataFrame]
    table_artifact: WandbArtifactConfig | None
    dataset_artifact: WandbArtifactConfig | None
    dataset_path: Path | None
