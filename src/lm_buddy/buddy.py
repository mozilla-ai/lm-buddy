from lm_buddy.integrations.wandb import ArtifactLoader, WandbArtifactLoader
from lm_buddy.jobs._entrypoints import run_finetuning, run_lm_harness, run_prometheus, run_ragas
from lm_buddy.jobs.common import EvaluationResult, FinetuningResult
from lm_buddy.jobs.configs import (
    EvaluationJobConfig,
    FinetuningJobConfig,
    LMHarnessJobConfig,
    PrometheusJobConfig,
    RagasJobConfig,
)


class LMBuddy:
    """Your buddy in the (L)LM space.

    Simple wrapper around executable functions for tasks available in the library.
    """

    def __init__(self, artifact_loader: ArtifactLoader = WandbArtifactLoader()):
        self._artifact_loader = artifact_loader

    def finetune(self, config: FinetuningJobConfig) -> FinetuningResult:
        """Run a supervised finetuning task with the provided configuration."""
        finetuning_result = run_finetuning(config, self._artifact_loader)
        return finetuning_result

    def evaluate(self, config: EvaluationJobConfig) -> EvaluationResult:
        """Run an evaluation task with the provided configuration.

        The underlying evaluation framework is determined by the configuration type.
        """
        match config:
            case LMHarnessJobConfig() as lm_harness_config:
                return run_lm_harness(lm_harness_config, self._artifact_loader)
            case PrometheusJobConfig() as prometheus_config:
                return run_prometheus(prometheus_config, self._artifact_loader)
            case RagasJobConfig() as ragas_config:
                return run_ragas(ragas_config, self._artifact_loader)
            case _:
                raise ValueError(f"Invlid configuration for evaluation: {type(config)}")
