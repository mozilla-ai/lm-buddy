from pydantic import validator

from flamingo.integrations.huggingface import HuggingFaceRepoConfig, convert_to_repo_config
from flamingo.integrations.wandb import WandbArtifactConfig
from flamingo.types import BaseFlamingoConfig


class AutoTokenizerConfig(BaseFlamingoConfig):
    """Settings passed to a HuggingFace AutoTokenizer instantiation."""

    load_from: HuggingFaceRepoConfig | WandbArtifactConfig
    trust_remote_code: bool | None = None
    use_fast: bool | None = None

    _validate_load_from = validator("load_from", pre=True, allow_reuse=True)(convert_to_repo_config)
