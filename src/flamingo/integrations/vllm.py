from flamingo.integrations.huggingface import HuggingFacePathReference
from flamingo.types import BaseFlamingoConfig


class InferenceServerConfig(BaseFlamingoConfig):
    """Generic configuration for an externally hosted inference endpoint.

    The inference server is defined by an endpoint with the provided `base_url`.
    The model `engine` powering the endpoint can also be specified
    to provide further metadata about the model used for inference
    (e.g., for generating W&B artifact lineages).

    Note: This configuration is intended to be generic and not bound to the interface
    of any specific training/evaluation framework. See `LocalChatCompletionConfig`
    for intended usage alongside a third-party framework.
    """

    base_url: str
    engine: str | HuggingFacePathReference | None = None
