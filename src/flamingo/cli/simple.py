import click


@click.group(name="simple", help="Simple test job.")
def group():
    pass


@group.command("describe", help="Describe the configuration for a simple test job.")
def describe() -> None:
    from flamingo.jobs.simple import SimpleJobConfig

    schema_json = SimpleJobConfig.schema_json(indent=2)
    click.secho(schema_json)


@group.command("run", help="Run a simple test job.")
@click.option("--config", type=str)
def run(config: str) -> None:
    from flamingo.jobs.simple import SimpleJobConfig, run_simple

    config = SimpleJobConfig.from_yaml_file(config)
    run_simple(config)
