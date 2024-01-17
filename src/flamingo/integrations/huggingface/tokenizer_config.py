from typing import Any

from pydantic import validator

from flamingo.integrations.huggingface.utils import repo_id_validator
from flamingo.integrations.wandb import WandbArtifactLink
from flamingo.types import BaseFlamingoConfig


class AutoTokenizerConfig(BaseFlamingoConfig):
    """Settings passed to a HuggingFace AutoTokenizer instantiation."""

    path: str | WandbArtifactLink
    trust_remote_code: bool | None = None
    use_fast: bool | None = None

    _path_validator = validator("path", allow_reuse=True, pre=True)(repo_id_validator)

    def get_tokenizer_args(self) -> dict[str, Any]:
        args = dict(
            trust_remote_code=self.trust_remote_code,
            use_fast=self.use_fast,
        )
        # Only return non-None values so we get HuggingFace defaults when not specified
        return {k: v for k, v in args.items() if v is not None}
