import json

import click

from lm_buddy.jobs.finetuning import FinetuningJobConfig
from lm_buddy.jobs.lm_harness import LMHarnessJobConfig
from lm_buddy.jobs.simple import SimpleJobConfig


@click.group(name="schema", help="Get a job configuration schema.")
def group():
    pass


@group.command("simple", help="Schema for the simple test job configuration.")
def schema_simple() -> None:
    schema = SimpleJobConfig.model_json_schema()
    click.secho(json.dumps(schema, indent=2))


@group.command("finetuning", help="Schema for the finetuning job configuration.")
def schema_finetuning() -> None:
    schema = FinetuningJobConfig.model_json_schema()
    click.secho(json.dumps(schema, indent=2))


@group.command("lm-harness", help="Schema for the lm-harness job configuration.")
def schema_lm_harness() -> None:
    schema = LMHarnessJobConfig.model_json_schema()
    click.secho(json.dumps(schema, indent=2))
