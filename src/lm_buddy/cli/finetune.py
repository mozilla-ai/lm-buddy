import click

from lm_buddy import LMBuddy
from lm_buddy.jobs.configs import FinetuningJobConfig


@click.command(name="finetune", help="Run an LM Buddy finetuning job.")
@click.option("--config", type=str)
def command(config: str) -> None:
    config = FinetuningJobConfig.from_yaml_file(config)

    buddy = LMBuddy()
    buddy.finetune(config)
