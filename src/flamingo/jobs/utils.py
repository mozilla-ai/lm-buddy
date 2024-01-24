from enum import Enum


class FlamingoJobType(str, Enum):
    """Enumeration of logical job types runnable via the Flamingo."""

    PREPROCESSING = "preprocessing"
    FINETUNING = "finetuning"
    EVALUATION = "evaluation"
