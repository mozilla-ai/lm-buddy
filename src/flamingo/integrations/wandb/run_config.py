import os
import warnings

from pydantic import root_validator
from wandb.apis.public import Run
from wandb.util import random_string

from flamingo.types import BaseFlamingoConfig


class WandbRunConfig(BaseFlamingoConfig):
    """Configuration required to log to a W&B run.

    A W&B Run is uniquely identified by the combination of `entity/project/run_id`.
    The W&B platform will auto-generate values for these variables if they are not provided
    when you initialize a run.

    However, based on how these attributes are passed between jobs it is often necessary
    to know the run ID before initializing a run.
    For this reason, the run ID field is made non-optional and auto-generated locally
    if it is not provided.
    """

    __match_args__ = ("run_id", "name", "project", "run_group", "entity")

    run_id: str
    name: str | None = None
    project: str | None = None
    run_group: str | None = None
    entity: str | None = None

    @root_validator(pre=True)
    def warn_missing_api_key(cls, values):
        if not os.environ.get("WANDB_API_KEY", None):
            warnings.warn(
                "Cannot find `WANDB_API_KEY` in your environment. "
                "Tracking will fail if a default key does not exist on the Ray cluster."
            )
        return values

    @root_validator(pre=True)
    def ensure_run_id(cls, values):
        if values.get("run_id", None) is None:
            # Generate an random 8-digit alphanumeric string, analogous to W&B platform
            values["run_id"] = random_string(length=8)
        return values

    @classmethod
    def from_run(cls, run: Run) -> "WandbRunConfig":
        """Extract environment settings from a W&B Run object.

        Useful when listing runs from the W&B API and extracting their settings for a job.
        """
        # TODO: Can we get the run group from this when it exists?
        return cls(
            name=run.name,
            project=run.project,
            entity=run.entity,
            run_id=run.id,
        )

    def wandb_path(self) -> str:
        """String identifier for the asset on the W&B platform."""
        path = "/".join(x for x in [self.entity, self.project, self.run_id] if x is not None)
        return path

    def env_vars(self) -> dict[str, str]:
        env_vars = {
            "WANDB_RUN_ID": self.run_id,
            "WANDB_NAME": self.name,
            "WANDB_PROJECT": self.project,
            "WANDB_RUN_GROUP": self.run_group,
            "WANDB_ENTITY": self.entity,
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", None),
        }
        return {k: v for k, v in env_vars.items() if v is not None}
