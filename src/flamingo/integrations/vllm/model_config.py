from flamingo.types import BaseFlamingoConfig


class InferenceServerConfig(BaseFlamingoConfig):
    """Inference Server URL endpoint path"""

    base_url: str