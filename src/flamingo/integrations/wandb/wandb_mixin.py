from flamingo.integrations.wandb import WandbEnvironment
from flamingo.types import BaseFlamingoConfig


class WandbEnvironmentMixin(BaseFlamingoConfig):
    """Mixin for a config that contains W&B environment settings."""

    wandb_env: WandbEnvironment | None = None

    @property
    def env_vars(self) -> dict[str, str]:
        return self.wandb_env.env_vars if self.wandb_env else {}

    @property
    def wandb_name(self) -> str | None:
        """Return the W&B run name, if it exists."""
        return self.wandb_env.name if self.wandb_env else None

    @property
    def wandb_project(self) -> str | None:
        """Return the W&B project name, if it exists."""
        return self.wandb_env.project if self.wandb_env else None
