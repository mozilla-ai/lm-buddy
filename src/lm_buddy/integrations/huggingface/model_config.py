from lm_buddy.paths import LoadableAssetPath
from lm_buddy.types import BaseLMBuddyConfig, SerializableTorchDtype


class AutoModelConfig(BaseLMBuddyConfig):
    """Settings passed to a HuggingFace AutoModel instantiation.

    The model to load can either be a HuggingFace repo or an artifact reference on W&B.
    """

    path: LoadableAssetPath
    trust_remote_code: bool = False
    torch_dtype: SerializableTorchDtype | None = None
