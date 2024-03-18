import click

from lm_buddy import LMBuddy
from lm_buddy.jobs.configs import FinetuningJobConfig, LMHarnessJobConfig, PrometheusJobConfig

# TODO(RD2024-125): Collapse the run commands into `lm-buddy finetune` and `lm-buddy evaluate`
#   to match the methods on the `LMBuddy` class

buddy = LMBuddy()


@click.group(name="run", help="Run an LM Buddy job.")
def group():
    pass


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
    buddy.evaluate(config)
