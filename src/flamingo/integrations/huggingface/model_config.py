from pydantic import validator

from flamingo.integrations.huggingface import HuggingFaceRepoConfig, convert_to_repo_config
from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig, SerializableTorchDtype


class AutoModelConfig(BaseFlamingoConfig):
    """Settings passed to a HuggingFace AutoModel instantiation.

    The model path can either be a string corresponding to a HuggingFace repo ID,
    or an artifact link to a reference artifact on W&B.
    """

    load_from: HuggingFaceRepoConfig | WandbArtifactConfig
    trust_remote_code: bool = False
    torch_dtype: SerializableTorchDtype = None

    _validate_load_from = validator("load_from", pre=True, allow_reuse=True)(convert_to_repo_config)
