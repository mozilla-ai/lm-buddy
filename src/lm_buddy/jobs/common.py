from enum import Enum

from lm_buddy.types import BaseLMBuddyConfig


class LMBuddyJobConfig(BaseLMBuddyConfig):
    """Configuration that comprises the entire input to an LM Buddy job.

    Currently, there is a 1:1 mapping between job entrypoints and job config implementations,
    but this is not rigidly constrained by the interface.
    We may adjust this in the future or add further fuctionality to this class
    to differentiate it from a `BaseLMBuddyConfig`.
    """

    pass


class LMBuddyJobType(str, Enum):
    """Enumeration of logical job types runnable via the LM Buddy."""

    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"
