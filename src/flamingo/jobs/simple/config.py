from flamingo.jobs.base import BaseJobConfig


class SimpleJobConfig(BaseJobConfig):
    """Simple job submission config."""

    magic_number: int
