from pydantic import validator

from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig, TorchDtypeString
from flamingo.integrations.vllm import LocalServerConfig


class InferenceServerConfig(BaseFlamingoConfig):
    """Inference Server URL endpoint path"""

    load_from: LocalServerConfig | WandbArtifactConfig

    trust_remote_code: bool = False
    torch_dtype: TorchDtypeString | None = None

    _validate_load_from_string = validator("load_from", pre=True, allow_reuse=True)(
        convert_string_to_repo_config
    )