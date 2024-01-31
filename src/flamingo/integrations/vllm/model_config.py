from flamingo.types import BaseFlamingoConfig


class InferenceServerConfig(BaseFlamingoConfig):
    """Inference Server URL endpoint path"""

    load_from: HuggingFaceRepoConfig 
    base_url: str
