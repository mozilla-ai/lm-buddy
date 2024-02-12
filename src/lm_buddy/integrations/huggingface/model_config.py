from pydantic import field_validator

from lm_buddy.integrations.huggingface import HuggingFaceRepoConfig, convert_string_to_repo_config
from lm_buddy.integrations.wandb import WandbArtifactConfig
from lm_buddy.types import BaseLMBuddyConfig, SerializableTorchDtype


class AutoModelConfig(BaseLMBuddyConfig):
    """Settings passed to a HuggingFace AutoModel instantiation.

    The model to load can either be a HuggingFace repo or an artifact reference on W&B.
    """

    load_from: HuggingFaceRepoConfig | WandbArtifactConfig
    trust_remote_code: bool = False
    torch_dtype: SerializableTorchDtype | None = None

    _validate_load_from_string = field_validator("load_from", mode="before")(
        convert_string_to_repo_config
    )
