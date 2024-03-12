from lm_buddy.paths import HuggingFaceAssetPath
from lm_buddy.types import BaseLMBuddyConfig


class AutoTokenizerConfig(BaseLMBuddyConfig):
    """Settings passed to a HuggingFace AutoTokenizer instantiation."""

    path: HuggingFaceAssetPath
    trust_remote_code: bool | None = None
    use_fast: bool | None = None
