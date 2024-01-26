import click

from flamingo.cli import finetuning, lm_harness, simple


@click.group(name="Flamingo CLI", help="Entrypoints for the Flamingo.")
def cli():
    pass


cli.add_command(finetuning.group)
cli.add_command(lm_harness.group)
cli.add_command(simple.group)
