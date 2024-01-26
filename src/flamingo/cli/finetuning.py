import click


@click.group(name="finetuning", help="Supervised finetuning of LLMs.")
def group():
    pass


@group.command("describe", help="Describe the configuration for a finetuning job.")
def describe() -> None:
    from flamingo.jobs.finetuning import FinetuningJobConfig

    schema_json = FinetuningJobConfig.schema_json(indent=2)
    click.secho(schema_json)


@group.command("run", help="Run a HuggingFace LLM finetuning job.")
@click.option("--config", type=str)
def run(config: str) -> None:
    from flamingo.jobs.finetuning import FinetuningJobConfig, run_finetuning

    config = FinetuningJobConfig.from_yaml_file(config)
    run_finetuning(config)
