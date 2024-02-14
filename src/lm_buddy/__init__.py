from lm_buddy.integrations.wandb import ArtifactLoader, WandbArtifactLoader
from lm_buddy.jobs.common import LMBuddyJobConfig
from lm_buddy.jobs.finetuning import FinetuningJobConfig, run_finetuning
from lm_buddy.jobs.lm_harness import LMHarnessJobConfig, run_lm_harness
from lm_buddy.jobs.simple import SimpleJobConfig, run_simple


def run(
    config: LMBuddyJobConfig,
    *,
    artifact_loader: ArtifactLoader = WandbArtifactLoader(),
) -> None:
    match config:
        case SimpleJobConfig() as simple_config:
            run_simple(simple_config, artifact_loader)
        case FinetuningJobConfig() as finetuning_config:
            run_finetuning(finetuning_config, artifact_loader)
        case LMHarnessJobConfig() as lm_harness_config:
            run_lm_harness(lm_harness_config, artifact_loader)
        case _:
            raise ValueError(f"Received invalid job configuration: {config}")
