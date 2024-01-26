import click


@click.group(name="schema", help="Get a job configuration schema.")
def group():
    pass


@group.command("simple", help="Schema for the simple test job configuration.")
def schema_simple() -> None:
    from flamingo.jobs.simple import SimpleJobConfig

    schema_json = SimpleJobConfig.schema_json(indent=2)
    click.secho(schema_json)


@group.command("finetuning", help="Schema for the finetuning job configuration.")
def schema_finetuning() -> None:
    from flamingo.jobs.finetuning import FinetuningJobConfig

    schema_json = FinetuningJobConfig.schema_json(indent=2)
    click.secho(schema_json)


@group.command("lm-harness", help="Schema for the lm-harness job configuration.")
def schema_lm_harness() -> None:
    from flamingo.jobs.lm_harness import LMHarnessJobConfig

    schema_json = LMHarnessJobConfig.schema_json(indent=2)
    click.secho(schema_json)
