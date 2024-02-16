import click

import lm_buddy
from lm_buddy.jobs.configs import FinetuningJobConfig, LMHarnessJobConfig, SimpleJobConfig

# TODO(RD2024-125): We should probably collapse all these commands into a single CLI command
# - Need to figure out best way to polymorphically deserialize the job config classes
# - Do we just add type discriminators at the job config level?


@click.group(name="run", help="Run an LM Buddy job.")
def group():
    pass


@group.command("simple", help="Run the simple test job.")
@click.option("--config", type=str)
def run_simple(config: str) -> None:
    config = SimpleJobConfig.from_yaml_file(config)
    lm_buddy.run_job(config)


@group.command("finetuning", help="Run the HuggingFace LLM finetuning job.")
@click.option("--config", type=str)
def run_finetuning(config: str) -> None:
    config = FinetuningJobConfig.from_yaml_file(config)
    lm_buddy.run_job(config)


@group.command("lm-harness", help="Run the lm-harness evaluation job.")
@click.option("--config", type=str)
def run_lm_harness(config: str) -> None:
    config = LMHarnessJobConfig.from_yaml_file(config)
    lm_buddy.run_job(config)
