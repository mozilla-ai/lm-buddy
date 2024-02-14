from lm_buddy.jobs.common import LMBuddyJobConfig


class SimpleJobConfig(LMBuddyJobConfig):
    """Simple job submission config."""

    magic_number: int
