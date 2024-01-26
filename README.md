# flamingo

<p align="center">
    <img src="https://github.com/mozilla-ai/flamingo/blob/main/assets/flamingo.png" width="300">
</p>

## Getting started

### Minimum Python version

This library is developed with the same Python version as the Ray cluster
to avoid dependency/syntax errors when executing code remotely.
Currently, installation requires Python between `[3.10, 3.11)` to match the global
cluster environment (Ray cluster is running 3.10.8).

### Installation

This project is built using the [Poetry](https://python-poetry.org/docs/) build tool.
First, install Poetry in your local environment via
```
curl -sSL https://install.python-poetry.org | python3 - -y
```
or see the [installation guide](https://python-poetry.org/docs/#installation)
for more instructions.

Once Poetry is installed, you can install `flamingo` for development by running
```
poetry lock
poetry install
```
This will install an editable version of the package along with all of its dependency groups.

Poetry should recognize your active virtual environment during installation
If you have an active Conda environment, Poetry should recognize it during installation
and install the package dependencies there. 
This hasn't been explicitly tested with other virtual python environments, but will likely work.

Alternatively, you can use poetry's own environment by running
```
poetry lock
poetry env use python3.10
poetry install
```
where `python3.10` is your python interpreter.

The `pyproject.toml` file defines dependency groups for the logical job types in the package.
Individual dependency groups can be installed by running 
`poetry install --with <group1>,<group2>` or `poetry install --only <group>`.

See the [contributing](CONTRIBUTING.md) guide for more information on development workflows.

### Usage

`flamingo` exposes a simple CLI with a few commands, one for each Ray job type.
Jobs are expected to take as input a YAML configuration file 
that contains all necessary parameters/settings for the work.
See the `examples/configs` folder for examples of the configuration structure.

Once installed in your environment, usage is as follows:
```
# Simple test
flamingo run simple --config simple_config.yaml

# LLM finetuning
flamingo run finetuning --config finetuning_config.yaml

# LLM evaluation
flamingo run lm-harness --config lm_harness_config.yaml
```
When submitting a job to Ray, the above commands should be used as your job entrypoints.
