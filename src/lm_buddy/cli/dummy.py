import click
from loguru import logger

from lm_buddy.tracking.platform import finish


@click.command(name="dummy", help="Run a dummy job.")
@click.option("--config", type=str)
def command(config: str) -> None:
    logger.info("I'm a teapot")
    finish()
