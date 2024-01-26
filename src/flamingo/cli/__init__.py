import click

from flamingo.cli import run, schema


@click.group(name="Flamingo CLI", help="Entrypoints for the Flamingo.")
def cli():
    pass


cli.add_command(run.group)
cli.add_command(schema.group)
