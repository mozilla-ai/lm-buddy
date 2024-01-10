from flamingo.jobs.configs import BaseJobConfig


class SimpleJobConfig(BaseJobConfig):
    """A simple job to demonstrate the submission interface."""

    magic_number: int
