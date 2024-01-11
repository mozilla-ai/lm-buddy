import os
import warnings

from pydantic import Extra, root_validator
from wandb.apis.public import Run

from flamingo.types import BaseFlamingoConfig


class WandbEnvironment(BaseFlamingoConfig):
    """Settings required to log to a W&B run.

    The fields on this class map to the environment variables
    that are used to control the W&B logging locations.

    The `name` and `project` are required as they are the minimum information
    required to identify a run. The `name` is the human-readable name that appears in the W&B UI.
    `name` is different than the `run_id` which must be unique within a project.
    Although the `name` is not mandatorily unique, it is generally best practice to use a
    unique and descriptive name to later identify the run.
    """

    class Config:
        extra = Extra.forbid  # Error on extra kwargs

    __match_args__ = ("name", "project", "run_id", "run_group", "entity")

    name: str | None = None
    project: str | None = None
    run_id: str | None = None
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

    @property
    def env_vars(self) -> dict[str, str]:
        # WandB w/ HuggingFace is weird. You can specify the run name inline,
        # but the rest must be injected as environment variables
        env_vars = {
            "WANDB_NAME": self.name,
            "WANDB_PROJECT": self.project,
            "WANDB_RUN_ID": self.run_id,
            "WANDB_RUN_GROUP": self.run_group,
            "WANDB_ENTITY": self.entity,
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", None),
        }
        return {k: v for k, v in env_vars.items() if v is not None}

    @classmethod
    def from_run(cls, run: Run) -> "WandbEnvironment":
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
