import click


@click.group()
def main():
    pass


@main.group(name="run")
def run():
    pass


@run.command("simple")
@click.option("--config", type=str)
def run_simple(config: str) -> None:
    from flamingo.jobs.configs import SimpleJobConfig
    from flamingo.jobs.entrypoints import simple_entrypoint

    config = SimpleJobConfig.from_yaml_file(config)
    simple_entrypoint.main(config)


@run.command("finetuning")
@click.option("--config", type=str)
def run_finetuning(config: str) -> None:
    from flamingo.jobs.configs import FinetuningJobConfig
    from flamingo.jobs.entrypoints import finetuning_entrypoint

    config = FinetuningJobConfig.from_yaml_file(config)
    finetuning_entrypoint.main(config)


@run.command("ludwig")
@click.option("--config", type=str)
@click.option("--dataset", type=str)
def run_ludwig(config: str, dataset: str) -> None:
    from flamingo.jobs.entrypoints import ludwig_entrypoint

    ludwig_entrypoint.main(config, dataset)


@run.command("lm-harness")
@click.option("--config", type=str)
def run_lm_harness(config: str) -> None:
    from flamingo.jobs.configs import LMHarnessJobConfig
    from flamingo.jobs.entrypoints import lm_harness_entrypoint

    config = LMHarnessJobConfig.from_yaml_file(config)
    lm_harness_entrypoint.main(config)


if __name__ == "__main__":
    main()
