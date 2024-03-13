from pathlib import Path
from typing import Any

from pydantic.dataclasses import dataclass

from lm_buddy.integrations.wandb import WandbArtifactConfig


@dataclass
class TaskOutput:
    artifact: WandbArtifactConfig | None


@dataclass
class ModelOutput(TaskOutput):
    path: Path
    is_adapter: bool


@dataclass
class DatasetOutput(TaskOutput):
    path: Path


@dataclass
class EvaluationOutput(TaskOutput):
    results: dict[str, Any]  # TODO: How should this data be stored?
