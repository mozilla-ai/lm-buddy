
from starlette.requests import Request
import ray
from ray import serve
from transformers import pipeline

from lm_buddy.integrations.huggingface import HuggingFaceAssetLoader
from lm_buddy.integrations.wandb import ArtifactLoader
from lm_buddy.jobs.configs import FinetuningJobConfig
from lm_buddy.jobs.configs.ray_serve import RayServeConfig
from typing import Dict


@serve.deployment
class ModelDeployment():
    def __init__(self, model):
        self._model = pipeline(model)

    def __call__(self, request: Request) -> Dict:
        return self._model(request.query_params["text"])[0]


def serve(
    config: RayServeConfig,
    artifact_loader: ArtifactLoader,
):
    hf_loader = HuggingFaceAssetLoader(artifact_loader)
    model = hf_loader.load_pretrained_model(config.model)

    @serve.deployment(num_replicas=config.deployments.num_replicas,
                      ray_actor_options={"num_cpus": config.deployments.ray_actor_options.num_cpus,
                                         "num_gpus": config.deployments.ray_actor_options.num_gpus})

    model = ModelDeployment(model)

    model.bind()