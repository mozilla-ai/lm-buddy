import click

from lm_buddy.cli import run, schema


@click.group(name="lm-buddy CLI", help="Entrypoints for the lm-buddy.")
def cli():
    pass


cli.add_command(run.group)
cli.add_command(schema.group)
