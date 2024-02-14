import click

import lm_buddy
from lm_buddy.jobs.finetuning import FinetuningJobConfig
from lm_buddy.jobs.lm_harness import LMHarnessJobConfig
from lm_buddy.jobs.simple import SimpleJobConfig

# TODO: Do we collapse all these commands into a single CLI run command?
# Need to figure out how to polymorphically deserialize the config classes?


@click.group(name="run", help="Run an LM Buddy job.")
def group():
    pass


@group.command("simple", help="Run the simple test job.")
@click.option("--config", type=str)
def simple_command(config: str) -> None:
    config = SimpleJobConfig.from_yaml_file(config)
    lm_buddy.run(config)


@group.command("finetuning", help="Run the HuggingFace LLM finetuning job.")
@click.option("--config", type=str)
def finetuning_command(config: str) -> None:
    config = FinetuningJobConfig.from_yaml_file(config)
    lm_buddy.run(config)


@group.command("lm-harness", help="Run the lm-harness evaluation job.")
@click.option("--config", type=str)
def lm_harness_command(config: str) -> None:
    config = LMHarnessJobConfig.from_yaml_file(config)
    lm_buddy.run(config)
