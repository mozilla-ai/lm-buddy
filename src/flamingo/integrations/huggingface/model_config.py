from pydantic import validator

from flamingo.integrations.huggingface.utils import repo_id_validator
from flamingo.integrations.wandb import WandbArtifactLink
from flamingo.types import BaseFlamingoConfig, SerializableTorchDtype


class AutoModelConfig(BaseFlamingoConfig):
    """Settings passed to a HuggingFace AutoModel instantiation.

    The model path can either be a string corresponding to a HuggingFace repo ID,
    or an artifact link to a reference artifact on W&B.
    """

    path: str | WandbArtifactLink
    trust_remote_code: bool = False
    torch_dtype: SerializableTorchDtype = None

    _path_validator = validator("path", allow_reuse=True, pre=True)(repo_id_validator)
