import click

from lm_buddy.cli import run, schema


@click.group(name="LM Buddy CLI", help="Entrypoints for the LM Buddy.")
def cli():
    pass


cli.add_command(run.group)
cli.add_command(schema.group)
