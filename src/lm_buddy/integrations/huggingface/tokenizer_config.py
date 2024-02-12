from pydantic import field_validator

from lm_buddy.integrations.huggingface import HuggingFaceRepoConfig, convert_string_to_repo_config
from lm_buddy.integrations.wandb import WandbArtifactConfig
from lm_buddy.types import BaseLMBuddyConfig


class AutoTokenizerConfig(BaseLMBuddyConfig):
    """Settings passed to a HuggingFace AutoTokenizer instantiation."""

    load_from: HuggingFaceRepoConfig | WandbArtifactConfig
    trust_remote_code: bool | None = None
    use_fast: bool | None = None

    _validate_load_from_string = field_validator("load_from", mode="before")(
        convert_string_to_repo_config
    )
