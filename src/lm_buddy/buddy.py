from lm_buddy.integrations.wandb import ArtifactLoader, WandbArtifactLoader
from lm_buddy.jobs._entrypoints import run_finetuning, run_lm_harness, run_prometheus, run_simple
from lm_buddy.jobs.common import EvaluationOutput, FinetuningOutput, SimpleOutput
from lm_buddy.jobs.configs import (
    EvaluationJobConfig,
    FinetuningJobConfig,
    LMHarnessJobConfig,
    PrometheusJobConfig,
    SimpleJobConfig,
)


class LMBuddy:
    """Your buddy in the (L)LM space.

    Simple wrapper around executable functions for tasks available in the library.
    """

    def __init__(self, artifact_loader: ArtifactLoader = WandbArtifactLoader()):
        self._artifact_loader = artifact_loader

    def simple(self, config: SimpleJobConfig) -> SimpleOutput:
        """Run a simple test task."""
        return run_simple(config)

    def finetune(self, config: FinetuningJobConfig) -> FinetuningOutput:
        """Run a supervised finetuning task with the provided configuration."""
        return run_finetuning(config)

    def evaluate(self, config: EvaluationJobConfig) -> EvaluationOutput:
        """Run an evaluation task with the provided configuration.

        The underlying evaluation framework is determined by the configuration type.
        """
        match config:
            case LMHarnessJobConfig() as lm_harness_config:
                return run_lm_harness(lm_harness_config, self._artifact_loader)
            case PrometheusJobConfig() as prometheus_config:
                return run_prometheus(prometheus_config, self._artifact_loader)
            case _:
                raise ValueError(f"Invlid configuration for evaluation: {type(config)}")
