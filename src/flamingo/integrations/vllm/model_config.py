from pydantic import field_validator

from flamingo.integrations.huggingface import HuggingFaceRepoConfig
from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig


class InferenceServerConfig(BaseFlamingoConfig):
    """Inference Server URL endpoint path.

    The `base_url` is used to communicate with the server but information about the
    `model_backend` being hosted on the server is also required.
    This is necessary to access the model's tokenizer or artifact lineage information.

    The `model_backend` can be represented by either a HF repo or a W&B artifact.
    """

    base_url: str
    model_backend: HuggingFaceRepoConfig | WandbArtifactConfig

    @field_validator("model_backend", mode="before")
    def validate_model_backend(cls, x):
        if isinstance(x, str):
            return HuggingFaceRepoConfig(repo_id=x)
        return x
