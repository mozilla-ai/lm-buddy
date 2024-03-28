
from starlette.requests import Request
import ray
from ray import serve
from transformers import pipeline

from lm_buddy.integrations.huggingface import HuggingFaceAssetLoader
from lm_buddy.integrations.wandb import ArtifactLoader
from lm_buddy.jobs.configs import FinetuningJobConfig
from lm_buddy.jobs.configs.ray_serve import RayServeConfig
from typing import Dict
from ray.serve.handle import RayServeHandle

config = RayServeConfig

@serve.deployment(num_replicas=config.deployments.num_replicas,
                      ray_actor_options={"num_cpus": config.deployments.ray_actor_options.num_cpus,
                                         "num_gpus": config.deployments.ray_actor_options.num_gpus})
class Summarizer:
    def __init__(self, translator: RayServeHandle, artifact_loader: ArtifactLoader):
        # Load model
        hf_loader = HuggingFaceAssetLoader(artifact_loader)
        model = hf_loader.load_pretrained_model(config.model)
        self.model = pipeline("summarization", model=model)
        self.translator = translator
        self.min_length = 5
        self.max_length = 15

    def summarize(self, text: str) -> str:
        # Run inference
        model_output = self.model(
            text, min_length=self.min_length, max_length=self.max_length
        )

        # Post-process output to return only the summary text
        summary = model_output[0]["summary_text"]

        return summary

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        summary = self.summarize(english_text)

        translation_ref = await self.translator.translate.remote(summary)
        translation = await translation_ref

        return translation

    def reconfigure(self, config: Dict):
        self.min_length = config.get("min_length", 5)
        self.max_length = config.get("max_length", 15)



app = Summarizer.bind()