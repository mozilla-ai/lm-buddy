import click


@click.group(name="Flamingo CLI", help="Entrypoints for the Flamingo.")
def cli():
    pass


@cli.group(name="run")
def run():
    pass


@run.command("simple", help="Run a simple test job.")
@click.option("--config", type=str)
def run_simple(config: str) -> None:
    from flamingo.jobs.simple import SimpleJobConfig, run_simple

    config = SimpleJobConfig.from_yaml_file(config)
    run_simple(config)


@run.command("finetuning", help="Run a HuggingFace LLM finetuning job.")
@click.option("--config", type=str)
def run_finetuning(config: str) -> None:
    from flamingo.jobs.finetuning import FinetuningJobConfig, run_finetuning

    config = FinetuningJobConfig.from_yaml_file(config)
    run_finetuning(config)


@run.command("lm-harness", help="Run an lm-harness LLM evaluation job.")
@click.option("--config", type=str)
def run_lm_harness(config: str) -> None:
    from flamingo.jobs.lm_harness import LMHarnessJobConfig, run_lm_harness

    config = LMHarnessJobConfig.from_yaml_file(config)
    run_lm_harness(config)


if __name__ == "__main__":
    cli()
