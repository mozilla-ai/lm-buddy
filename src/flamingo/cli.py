import click

from flamingo.jobs.configs import FinetuningJobConfig, LMHarnessJobConfig, SimpleJobConfig
from flamingo.jobs.entrypoints import (
    finetuning_entrypoint,
    lm_harness_entrypoint,
    simple_entrypoint,
)


@click.group()
def main():
    pass


@main.group(name="run")
def run():
    pass


@run.command("simple")
@click.option("--config", type=str)
def run_simple(config: str) -> None:
    config = SimpleJobConfig.from_yaml_file(config)
    simple_entrypoint.main(config)


@run.command("finetuning")
@click.option("--config", type=str)
def run_finetuning(config: str) -> None:
    config = FinetuningJobConfig.from_yaml_file(config)
    finetuning_entrypoint.main(config)


@run.command("lm-harness")
@click.option("--config", type=str)
def run_lm_harness(config: str) -> None:
    config = LMHarnessJobConfig.from_yaml_file(config)
    lm_harness_entrypoint.main(config)


if __name__ == "__main__":
    main()
