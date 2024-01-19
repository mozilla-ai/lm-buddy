from enum import Enum


class ArtifactType(str, Enum):
    """Enumeration of artifact types used by the Flamingo."""

    DATASET = "dataset"
    MODEL = "model"
    TOKENIZER = "tokenizer"
    EVALUATION = "evaluation"
