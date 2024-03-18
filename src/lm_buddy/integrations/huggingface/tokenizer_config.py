from lm_buddy.paths import AssetPath
from lm_buddy.types import BaseLMBuddyConfig


class AutoTokenizerConfig(BaseLMBuddyConfig):
    """Settings passed to a HuggingFace AutoTokenizer instantiation."""

    load_from: AssetPath
    trust_remote_code: bool | None = None
    use_fast: bool | None = None
