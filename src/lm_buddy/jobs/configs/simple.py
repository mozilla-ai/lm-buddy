from lm_buddy.jobs.configs import LMBuddyJobConfig


class SimpleJobConfig(LMBuddyJobConfig):
    """Simple job submission config."""

    magic_number: int
