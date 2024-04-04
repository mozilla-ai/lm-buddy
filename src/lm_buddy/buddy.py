import wandb

from lm_buddy.integrations.wandb import WandbResumeMode
from lm_buddy.jobs._entrypoints import run_finetuning, run_lm_harness, run_prometheus, run_ragas
from lm_buddy.jobs.common import EvaluationResult, FinetuningResult, LMBuddyJobType
from lm_buddy.jobs.configs import (
    EvaluationJobConfig,
    FinetuningJobConfig,
    LMBuddyJobConfig,
    LMHarnessJobConfig,
    PrometheusJobConfig,
    RagasJobConfig,
)
from lm_buddy.paths import strip_path_prefix


class LMBuddy:
    """Your buddy in the (L)LM space.

    Simple wrapper around executable functions for tasks available in the library.
    """

    # TODO: Store some configuration (e.g., tracking info, name) globally on the buddy
    def __init__(self):
        pass

    def _generate_artifact_lineage(
        self,
        config: LMBuddyJobConfig,
        results: list[wandb.Artifact],
        job_type: LMBuddyJobType,
    ) -> None:
        """Link input artifacts and log output artifacts to a run.

        A no-op if no tracking config is available.
        """
        if config.tracking is not None:
            with wandb.init(
                name=config.name,
                job_type=job_type,
                resume=WandbResumeMode.ALLOW,
                **config.tracking.model_dump(),
            ) as run:
                for path in config.artifact_paths():
                    artifact_name = strip_path_prefix(path)
                    run.use_artifact(artifact_name)
                for artifact in results:
                    run.log_artifact(artifact)

    def finetune(self, config: FinetuningJobConfig) -> FinetuningResult:
        """Run a supervised finetuning job with the provided configuration."""
        result = run_finetuning(config)
        self._generate_artifact_lineage(config, result.artifacts, LMBuddyJobType.FINETUNING)
        return result

    def evaluate(self, config: EvaluationJobConfig) -> EvaluationResult:
        """Run an evaluation job with the provided configuration.

        The underlying evaluation framework is determined by the configuration type.
        """
        match config:
            case LMHarnessJobConfig() as lm_harness_config:
                result = run_lm_harness(lm_harness_config)
            case PrometheusJobConfig() as prometheus_config:
                result = run_prometheus(prometheus_config)
            case RagasJobConfig() as ragas_config:
                result = run_ragas(ragas_config)
            case _:
                raise ValueError(f"Invlid configuration for evaluation: {type(config)}")
        self._generate_artifact_lineage(config, result.artifacts, LMBuddyJobType.EVALUATION)
        return result
