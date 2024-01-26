import click


@click.group(name="lm-harness", help="LLM evaluation with the lm-evaluation-harness.")
def group():
    pass


@group.command("describe", help="Describe the configuration for an lm-harness job.")
def describe() -> None:
    from flamingo.jobs.lm_harness import LMHarnessJobConfig

    schema_json = LMHarnessJobConfig.schema_json(indent=2)
    click.secho(schema_json)


@group.command("run", help="Run an lm-harness evaluation job.")
@click.option("--config", type=str)
def run(config: str) -> None:
    from flamingo.jobs.lm_harness import LMHarnessJobConfig, run_lm_harness

    config = LMHarnessJobConfig.from_yaml_file(config)
    run_lm_harness(config)
