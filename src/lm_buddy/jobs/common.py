from enum import Enum
from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass

from lm_buddy.integrations.wandb import WandbArtifactConfig


class LMBuddyJobType(str, Enum):
    """Enumeration of logical job types runnable via the LM Buddy."""

    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class JobOutput:
    artifact: WandbArtifactConfig | None


@dataclass
class ModelOutput(JobOutput):
    path: Path
    is_adapter: bool


@dataclass
class EvaluationOutput(JobOutput):
    results: dict[str, Any]
