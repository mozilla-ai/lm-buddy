from pydantic import validator

from flamingo.integrations.huggingface import HuggingFaceRepoConfig, convert_string_to_repo_config
from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig, SerializableTorchDtype


class AutoModelConfig(BaseFlamingoConfig):
    """Settings passed to a HuggingFace AutoModel instantiation.

    The model to load can either be a HuggingFace repo or an artifact reference on W&B.
    """

    load_from: HuggingFaceRepoConfig | WandbArtifactConfig
    trust_remote_code: bool = False
    torch_dtype: SerializableTorchDtype = None

    _validate_load_from_string = validator("load_from", pre=True, allow_reuse=True)(
        convert_string_to_repo_config
    )
