import click

from lm_buddy import LMBuddy
from lm_buddy.jobs.configs import LMHarnessJobConfig, PrometheusJobConfig


@click.group(name="evaluate", help="Run an LM Buddy evaluation job.")
def group() -> None:
    pass


@group.command("lm-harness", help="Run the lm-harness evaluation job.")
@click.option("--config", type=str)
def lm_harness_command(config: str) -> None:
    config = LMHarnessJobConfig.from_yaml_file(config)

    buddy = LMBuddy()
    buddy.evaluate(config)


@group.command("prometheus", help="Run the prometheus evaluation job.")
@click.option("--config", type=str)
def prometheus_command(config: str) -> None:
    config = PrometheusJobConfig.from_yaml_file(config)

    buddy = LMBuddy()
    buddy.evaluate(config)
