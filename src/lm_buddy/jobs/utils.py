from enum import Enum


class LMBuddyJobType(str, Enum):
    """Enumeration of logical job types runnable via the lm-buddy."""

    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"
