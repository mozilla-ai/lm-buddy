from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from lm_buddy.paths import AssetPath


class LMBuddyJobType(str, Enum):
    """Enumeration of logical job types runnable via the LM Buddy."""

    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class FinetuningResult:
    """Result from a finetuning task."""

    checkpoint_path: AssetPath | None
    checkpoint_artifact_path: AssetPath | None
    metrics: dict[str, Any]
    is_adapter: bool


@dataclass
class EvaluationResult:
    """Result from an evaluation task, containing aggregate metrics and artifacts."""

    tables: dict[str, pd.DataFrame]
    table_artifact_path: AssetPath | None
    dataset_path: AssetPath | None
    dataset_artifact_path: AssetPath | None
