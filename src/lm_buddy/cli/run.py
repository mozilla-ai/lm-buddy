import click

from lm_buddy.integrations.wandb import WandbArtifactLoader


@click.group(name="run", help="Run an lm-buddy job.")
def group():
    pass


@group.command("simple", help="Run the simple test job.")
@click.option("--config", type=str)
def run_simple(config: str) -> None:
    from lm_buddy.jobs.simple import SimpleJobConfig, run_simple

    config = SimpleJobConfig.from_yaml_file(config)
    run_simple(config)


@group.command("finetuning", help="Run the HuggingFace LLM finetuning job.")
@click.option("--config", type=str)
def run_finetuning(config: str) -> None:
    from lm_buddy.jobs.finetuning import FinetuningJobConfig, run_finetuning

    config = FinetuningJobConfig.from_yaml_file(config)
    artifact_loader = WandbArtifactLoader()
    run_finetuning(config, artifact_loader)


@group.command("lm-harness", help="Run the lm-harness evaluation job.")
@click.option("--config", type=str)
def run_lm_harness(config: str) -> None:
    from lm_buddy.jobs.lm_harness import LMHarnessJobConfig, run_lm_harness

    config = LMHarnessJobConfig.from_yaml_file(config)
    artifact_loader = WandbArtifactLoader()
    run_lm_harness(config, artifact_loader)
