from enum import Enum
from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass

from lm_buddy.integrations.wandb import WandbArtifactConfig


class LMBuddyJobType(str, Enum):
    """Enumeration of logical job types runnable via the LM Buddy."""

    SIMPLE = "simple"
    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"


@dataclass
class SimpleOutput:
    magic_number: int


@dataclass
class FinetuningOutput:
    artifact: WandbArtifactConfig | None
    checkpoint_path: Path | None
    is_adapter: bool


@dataclass
class EvaluationOutput:
    artifact: WandbArtifactConfig | None
    results: dict[str, Any]
