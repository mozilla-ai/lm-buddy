import click

from lm_buddy import LMBuddy
from lm_buddy.jobs.configs import FinetuningJobConfig
from lm_buddy.jobs.configs.ray_serve import RayServeAppConfig


@click.command(name="serve", help="Run a Ray Serve job.")
@click.option("--config", type=str)
def command(config: str) -> None:
    config = RayServeAppConfig.from_yaml_file(config)

    buddy = LMBuddy()
    buddy.serve(config)
