from lm_buddy.integrations.wandb import ArtifactLoader, WandbArtifactLoader
from lm_buddy.jobs._entrypoints import run_finetuning, run_lm_harness, run_prometheus, run_simple
from lm_buddy.jobs.common import JobOutput
from lm_buddy.jobs.configs import (
    EvaluationJobConfig,
    FinetuningJobConfig,
    LMHarnessJobConfig,
    PrometheusJobConfig,
)
from lm_buddy.jobs.configs.simple import SimpleJobConfig


class LMBuddy:
    """Your buddy in the (L)LM space.

    Simple wrapper around executable functions for tasks available in the library.
    """

    def __init__(self, artifact_loader: ArtifactLoader = WandbArtifactLoader()):
        self._artifact_loader = artifact_loader

    def simple(self, config: SimpleJobConfig) -> list[JobOutput]:
        return run_simple(config)

    def finetune(self, config: FinetuningJobConfig) -> list[JobOutput]:
        return run_finetuning(config)

    def evaluate(self, config: EvaluationJobConfig) -> list[JobOutput]:
        match config:
            case LMHarnessJobConfig() as lm_harness_config:
                return run_lm_harness(lm_harness_config, self._artifact_loader)
            case PrometheusJobConfig() as prometheus_config:
                return run_prometheus(prometheus_config, self._artifact_loader)
            case _:
                raise ValueError(f"Invlid configuration for evaluation: {type(config)}")
