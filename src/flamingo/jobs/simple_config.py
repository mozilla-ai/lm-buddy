from flamingo.jobs import BaseJobConfig


class SimpleJobConfig(BaseJobConfig):
    """A simple job to demonstrate the submission interface."""

    magic_number: int

    @property
    def env_vars(self) -> dict[str, str]:
        return {}

    @property
    def entrypoint_command(self) -> str:
        return f"python simple.py --magic_number '{self.magic_number}'"
