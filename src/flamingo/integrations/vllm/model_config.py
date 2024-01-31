from flamingo.types import BaseFlamingoConfig


class InferenceServerConfig(BaseFlamingoConfig):
    """Inference Server URL endpoint path"""

    base_url: str
    tokenizer: str
    model_name: str
    tokenizer_backend: str
