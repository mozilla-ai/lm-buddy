import click

import lm_buddy
from lm_buddy.buddy import LMBuddy
from lm_buddy.integrations.wandb import WandbArtifactLoader
from lm_buddy.jobs.configs import (
    FinetuningJobConfig,
    LMHarnessJobConfig,
    PrometheusJobConfig,
    SimpleJobConfig,
)

buddy = LMBuddy(artifact_loader=WandbArtifactLoader())


@click.group(name="run", help="Run an LM Buddy job.")
def group():
    pass


@group.command("simple", help="Run the simple test job.")
@click.option("--config", type=str)
def run_simple(config: str) -> None:
    config = SimpleJobConfig.from_yaml_file(config)
    buddy.finetune(config)


@group.command("finetuning", help="Run the HuggingFace LLM finetuning job.")
@click.option("--config", type=str)
def run_finetuning(config: str) -> None:
    config = FinetuningJobConfig.from_yaml_file(config)
    buddy.finetune(config)


@group.command("lm-harness", help="Run the lm-harness evaluation job.")
@click.option("--config", type=str)
def run_lm_harness(config: str) -> None:
    config = LMHarnessJobConfig.from_yaml_file(config)
    buddy.evaluate(config)


@group.command("prometheus", help="Run the prometheus evaluation job.")
@click.option("--config", type=str)
def run_prometheus(config: str) -> None:
    config = PrometheusJobConfig.from_yaml_file(config)
    lm_buddy.run_job(config)
