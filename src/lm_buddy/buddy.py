import wandb

from lm_buddy.integrations.wandb import WandbResumeMode, wandb_init_from_config
from lm_buddy.jobs._entrypoints import run_finetuning, run_lm_harness, run_prometheus, run_ragas
from lm_buddy.jobs.common import EvaluationResult, FinetuningResult, LMBuddyJobType
from lm_buddy.jobs.configs import (
    EvaluationJobConfig,
    FinetuningJobConfig,
    LMHarnessJobConfig,
    PrometheusJobConfig,
    RagasJobConfig,
)
from lm_buddy.jobs.configs.base import LMBuddyJobConfig


class LMBuddy:
    """Your buddy in the (L)LM space.

    Simple wrapper around executable functions for tasks available in the library.
    """

    def __init__(self, *, offline: bool = False):
        self._offline = offline

    def _generate_artifact_lineage(
        self,
        config: LMBuddyJobConfig,
        result_artifacts: list[wandb.Artifact],
        job_type: LMBuddyJobType,
    ) -> None:
        if config.tracking is not None:
            with wandb_init_from_config(
                config.tracking,
                job_type=job_type,
                resume=WandbResumeMode.ALLOW,
            ) as run:
                for name in config.artifact_names():
                    run.use_artifact(name)
                for artifact in result_artifacts:
                    run.log_artifact(artifact)

    def finetune(self, config: FinetuningJobConfig) -> FinetuningResult:
        """Run a supervised finetuning task with the provided configuration."""
        result = run_finetuning(config, self._artifact_loader)

        if not self._offline:
            self._generate_artifact_lineage(config, result.artifacts, LMBuddyJobType.FINETUNING)

        return result

    def evaluate(self, config: EvaluationJobConfig) -> EvaluationResult:
        """Run an evaluation task with the provided configuration.

        The underlying evaluation framework is determined by the configuration type.
        """
        match config:
            case LMHarnessJobConfig() as lm_harness_config:
                result = run_lm_harness(lm_harness_config, self._artifact_loader)
            case PrometheusJobConfig() as prometheus_config:
                result = run_prometheus(prometheus_config, self._artifact_loader)
            case RagasJobConfig() as ragas_config:
                result = run_ragas(ragas_config, self._artifact_loader)
            case _:
                raise ValueError(f"Invlid configuration for evaluation: {type(config)}")

        if not self._offline:
            self._generate_artifact_lineage(config, result.artifacts, LMBuddyJobType.EVALUATION)

        return result
