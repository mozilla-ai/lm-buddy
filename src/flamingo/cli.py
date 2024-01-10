import click

from flamingo.jobs import run_finetuning, run_lm_harness, run_simple
from flamingo.jobs.configs import FinetuningJobConfig, LMHarnessJobConfig, SimpleJobConfig


@click.group()
def cli():
    pass


@click.group("simple")
@click.option("--config", type=str)
def run_simple_cli(config: str) -> None:
    config = SimpleJobConfig.from_yaml_file(config)
    run_simple.main(config)


@click.group("finetune")
@click.option("--config", type=str)
def run_finetuning_cli(config: str) -> None:
    config = FinetuningJobConfig.from_yaml_file(config)
    run_finetuning.main(config)


@click.group("finetune")
@click.option("--config", type=str)
def run_finetuning_cli(config: str) -> None:
    config = FinetuningJobConfig.from_yaml_file(config)
    run_finetuning.main(config)


# need to add the group / command function itself, not the module
cli.add_command(run_simple.main)
cli.add_command(run_finetuning.main)
cli.add_command(run_lm_harness.main)


if __name__ == "__main__":
    cli()
